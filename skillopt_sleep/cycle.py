"""SkillOpt-Sleep — the nightly cycle orchestrator.

run_sleep_cycle() wires the stages:
    harvest -> mine -> replay -> consolidate(gate) -> stage  (-> optional adopt)

It is pure-Python and import-light; with backend="mock" it runs with no API
key and no third-party deps, which is what the deterministic experiment and
CI use. With backend="anthropic" it spends the user's budget for real lift.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from skillopt_sleep.backend import get_backend
from skillopt_sleep.config import SleepConfig, load_config
from skillopt_sleep.dream import dream_consolidate
from skillopt_sleep.harvest_sources import harvest_for_config
from skillopt_sleep.memory import ensure_skill_scaffold
from skillopt_sleep.mine import count_checkable_tasks, mine
from skillopt_sleep.staging import adopt as adopt_staging
from skillopt_sleep.staging import write_staging
from skillopt_sleep.state import SleepState, _now_iso
from skillopt_sleep.types import SessionDigest, SleepReport, TaskRecord


@dataclass
class CycleOutcome:
    report: SleepReport
    staging_dir: str
    adopted: bool
    adopted_paths: List[str]


def _project_paths(cfg: SleepConfig) -> str:
    """Where live CLAUDE.md lives + which project we are evolving."""
    if cfg.get("projects") == "invoked" and cfg.get("invoked_project"):
        return cfg.get("invoked_project")
    # default: the invoked cwd
    return cfg.get("invoked_project") or os.getcwd()


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _progress(cfg: SleepConfig, message: str) -> None:
    if cfg.get("progress", False):
        print(f"[sleep] {message}", file=sys.stderr, flush=True)


def _default_miner_diagnostics() -> Dict[str, Any]:
    return {
        "miner_mode": "",
        "llm_miner_attempted": False,
        "sessions_passed_to_miner": 0,
        "llm_tasks_returned": 0,
        "llm_parse_failures": 0,
        "llm_empty_responses": 0,
        "llm_backend_errors": 0,
        "llm_miner_error": "",
        "fallback_used": False,
        "n_checkable_tasks": 0,
        "n_checkable_val_tasks": 0,
    }


def _apply_miner_diagnostics(report: SleepReport, diagnostics: Dict[str, Any]) -> None:
    report.miner_mode = str(diagnostics.get("miner_mode") or "")
    report.llm_miner_attempted = bool(diagnostics.get("llm_miner_attempted", False))
    report.sessions_passed_to_miner = int(diagnostics.get("sessions_passed_to_miner", 0) or 0)
    report.llm_tasks_returned = int(diagnostics.get("llm_tasks_returned", 0) or 0)
    report.llm_parse_failures = int(diagnostics.get("llm_parse_failures", 0) or 0)
    report.llm_empty_responses = int(diagnostics.get("llm_empty_responses", 0) or 0)
    report.llm_backend_errors = int(diagnostics.get("llm_backend_errors", 0) or 0)
    report.llm_miner_error = str(diagnostics.get("llm_miner_error") or "")
    report.fallback_used = bool(diagnostics.get("fallback_used", False))
    report.n_checkable_tasks = int(diagnostics.get("n_checkable_tasks", 0) or 0)
    report.n_checkable_val_tasks = int(diagnostics.get("n_checkable_val_tasks", 0) or 0)


def _checkable_gate_reason(cfg: SleepConfig, backend_name: str, tasks: List[TaskRecord]) -> str:
    if backend_name == "mock":
        return ""
    min_total = int(cfg.get("min_checkable_tasks", 3) or 0)
    min_val = int(cfg.get("min_checkable_val_tasks", 2) or 0)
    if count_checkable_tasks(tasks) < min_total:
        return "no_checkable_validation_tasks"
    if count_checkable_tasks(tasks, split="val") < min_val:
        return "no_checkable_validation_tasks"
    return ""


def _finish_without_consolidation(
    cfg: SleepConfig,
    *,
    report: SleepReport,
    state: SleepState,
    project: str,
    started: str,
    live_skill_path: str,
    live_memory_path: str,
    tasks: List[TaskRecord],
    dry_run: bool,
    clock: Optional[float],
    tokens_used: int,
) -> CycleOutcome:
    report.n_replayed = 0
    report.tokens_used = tokens_used
    report.ended_at = _now_iso(clock)
    staging_dir = ""
    if not dry_run:
        if tasks:
            state.add_to_archive([t.to_dict() for t in tasks if t.origin != "dream"])
        report_md = _render_report_md(report, cfg)
        staging_dir = write_staging(
            project,
            report=report,
            proposed_skill=None,
            proposed_memory=None,
            live_skill_path=live_skill_path,
            live_memory_path=live_memory_path,
            report_md=report_md,
        )
        state.set_last_harvest(project, started)
        state.record_night({
            "night": report.night,
            "accepted": False,
            "gate_action": report.gate_action,
            "no_edits_reason": report.no_edits_reason,
            "n_tasks": report.n_tasks,
            "n_checkable_tasks": report.n_checkable_tasks,
            "n_checkable_val_tasks": report.n_checkable_val_tasks,
            "staging": staging_dir,
        })
        state.save()
    return CycleOutcome(report, staging_dir, False, [])


def _render_report_md(report: SleepReport, cfg: SleepConfig) -> str:
    lines = [
        f"# SkillOpt-Sleep — night {report.night} report",
        "",
        f"- project: `{report.project}`",
        f"- backend: `{cfg.get('backend')}`  replay: `{cfg.get('replay_mode')}`",
        f"- sessions harvested: {report.n_sessions}",
        f"- tasks mined: {report.n_tasks}  (replayed: {report.n_replayed})",
        f"- gate: **{report.gate_action}** (accepted={report.accepted})",
        f"- tokens used: {report.tokens_used}",
        "",
    ]
    if report.n_replayed:
        lines.insert(6, f"- held-out score: {report.baseline_score:.3f} -> {report.candidate_score:.3f}")
    if report.no_edits_reason or report.pre_gate_status:
        lines.extend([
            "## Skip reason",
            f"- reason: `{report.no_edits_reason or report.pre_gate_status}`",
            "",
        ])
    lines.extend([
        "## Miner diagnostics",
        f"- miner mode: `{report.miner_mode or 'unknown'}`",
        f"- LLM miner attempted: {report.llm_miner_attempted}",
        f"- sessions passed to miner: {report.sessions_passed_to_miner}",
        f"- LLM tasks returned: {report.llm_tasks_returned}",
        f"- parse failures: {report.llm_parse_failures}",
        f"- empty responses: {report.llm_empty_responses}",
        f"- backend errors: {report.llm_backend_errors}",
        f"- fallback used: {report.fallback_used}",
        f"- checkable tasks: {report.n_checkable_tasks}",
        f"- checkable val tasks: {report.n_checkable_val_tasks}",
        "",
    ])
    if report.llm_miner_error:
        lines.extend([
            "## Miner error",
            f"- `{report.llm_miner_error}`",
            "",
        ])
    if report.edits:
        lines.append("## Accepted edits")
        for e in report.edits:
            lines.append(f"- [{e.target}/{e.op}] {e.content}  \n  _why: {e.rationale}_")
        lines.append("")
    if report.rejected_edits:
        lines.append("## Rejected by gate (kept as negative feedback)")
        for e in report.rejected_edits:
            lines.append(f"- [{e.target}/{e.op}] {e.content}")
        lines.append("")
    if report.notes:
        lines.append("## Notes")
        for n in report.notes:
            lines.append(f"- {n}")
        lines.append("")
    if report.accepted and (report.edits or report.rejected_edits):
        lines.append("_Review, then run `/sleep adopt` to apply, or discard this folder._")
    else:
        lines.append("_Review diagnostics; no proposal was generated for adoption._")
    return "\n".join(lines)


def run_sleep_cycle(
    cfg: Optional[SleepConfig] = None,
    *,
    seed_tasks: Optional[List[TaskRecord]] = None,
    dry_run: bool = False,
    clock: Optional[float] = None,
) -> CycleOutcome:
    """Run one full sleep cycle and return the outcome.

    Parameters
    ----------
    cfg : SleepConfig
    seed_tasks : optional pre-built TaskRecords (used by the experiment to
        inject a known persona instead of harvesting ~/.claude).
    dry_run : harvest+mine+replay but DO NOT stage/adopt (report only).
    clock : fixed epoch seconds for deterministic timestamps in tests.
    """
    cfg = cfg or load_config()
    state = SleepState.load(cfg.state_path)
    night = state.begin_night(clock)
    project = _project_paths(cfg)
    started = _now_iso(clock)
    backend_name = cfg.get("backend", "mock")

    backend = get_backend(
        backend_name,
        model=cfg.get("model", ""),
        codex_path=cfg.get("codex_path", ""),
        project_dir=project,
    )
    _progress(cfg, f"night {night}: project={project} backend={backend.name}")

    # ── live skill/memory docs ───────────────────────────────────────────
    live_memory_path = os.path.join(project, "CLAUDE.md")
    live_skill_path = cfg.managed_skill_path()
    _progress(cfg, f"live skill: {live_skill_path}")
    raw_skill = _read(live_skill_path)
    skill = raw_skill
    memory = _read(live_memory_path)
    if not skill:
        skill = ensure_skill_scaffold(
            "", name=cfg.get("managed_skill_name", "skillopt-sleep-learned"),
            description="Preferences and procedures learned from past local agent sessions.",
        )
    target_filter = bool(
        cfg.get("target_task_filter", True)
        and cfg.get("target_skill_path", "")
        and raw_skill
    )

    # ── 1+2. harvest + mine (unless seed_tasks injected) ─────────────────
    digests: List[SessionDigest] = []
    miner_diagnostics = _default_miner_diagnostics()
    if seed_tasks is not None:
        tasks = seed_tasks
        n_sessions = 0
        miner_diagnostics["miner_mode"] = cfg.get("task_source_mode", "seed_tasks")
        miner_diagnostics["n_checkable_tasks"] = count_checkable_tasks(tasks)
        miner_diagnostics["n_checkable_val_tasks"] = count_checkable_tasks(tasks, split="val")
        _progress(cfg, f"using {len(tasks)} seeded tasks")
    else:
        since = state.last_harvest_for(project)
        # On first run (no prior harvest), apply lookback_hours so we don't
        # scan the entire transcript history and trigger massive LLM mining.
        if since is None:
            lookback_hours = cfg.get("lookback_hours", 72)
            if lookback_hours is not None and lookback_hours > 0:
                import time
                ref_time = clock if clock is not None else time.time()
                cutoff = ref_time - lookback_hours * 3600
                since = _now_iso(cutoff)
        max_tasks = cfg.get("max_tasks_per_night", 40)
        max_sessions = cfg.get("max_sessions_per_night", 0) or max_tasks * 3
        candidate_limit = max_tasks
        if target_filter:
            candidate_limit = max(max_tasks, max_tasks * 3)
        _progress(
            cfg,
            f"harvest start: source={cfg.get('transcript_source')} max_sessions={max_sessions}",
        )
        digests = harvest_for_config(
            cfg,
            since_iso=since,
            limit=max_sessions,
        )
        n_sessions = len(digests)
        _progress(cfg, f"harvest done: sessions={n_sessions}")
        # Real backends must mine checkable tasks. The heuristic fallback is
        # useful for mock/debug, but it creates reference_kind=none tasks that
        # cannot provide a trustworthy validation gate for nightly optimization.
        llm_miner = None
        real_backend = backend_name != "mock"
        allow_fallback = (not real_backend) or bool(cfg.get("allow_uncheckable_fallback", False))
        if real_backend and cfg.get("llm_mine", True):
            try:
                from skillopt_sleep.llm_miner import make_llm_miner
                llm_miner = make_llm_miner(
                    backend,
                    max_sessions=max_sessions,
                    max_tasks=candidate_limit,
                    diagnostics=miner_diagnostics,
                )
            except Exception as exc:
                miner_diagnostics["llm_miner_error"] = f"{type(exc).__name__}: {exc}"
                llm_miner = None
        elif real_backend:
            miner_diagnostics["llm_miner_error"] = "llm_mine disabled for real backend"
        _progress(
            cfg,
            f"mine start: max_tasks={max_tasks} candidate_limit={candidate_limit} "
            f"llm_mine={llm_miner is not None} target_filter={target_filter}",
        )
        if real_backend and llm_miner is None and not allow_fallback:
            tasks = []
            miner_diagnostics["miner_mode"] = "llm_unavailable"
        else:
            tasks = mine(
                digests,
                max_tasks=max_tasks,
                candidate_limit=candidate_limit,
                holdout_fraction=cfg.get("holdout_fraction", 0.34),
                seed=cfg.get("seed", 42),
                llm_miner=llm_miner,
                allow_heuristic_fallback=allow_fallback,
                diagnostics=miner_diagnostics,
                target_skill_text=raw_skill if target_filter else "",
                target_skill_path=live_skill_path if target_filter else "",
            )
        _progress(cfg, f"mine done: tasks={len(tasks)}")

    report = SleepReport(
        night=night, project=project, started_at=started,
        n_sessions=n_sessions, n_tasks=len(tasks),
    )
    _apply_miner_diagnostics(report, miner_diagnostics)

    if not tasks:
        report.gate_action = "skip"
        report.no_edits_reason = (
            "no_checkable_validation_tasks" if backend_name != "mock" else "no_tasks_mined"
        )
        report.pre_gate_status = report.no_edits_reason
        report.notes.append("no tasks mined; replay/consolidation skipped")
        if backend_name != "mock":
            report.notes.append(
                "real-backend heuristic fallback is disabled; use LLM-mined "
                "checkable tasks or a reviewed --tasks-file"
            )
        return _finish_without_consolidation(
            cfg,
            report=report,
            state=state,
            project=project,
            started=started,
            live_skill_path=live_skill_path,
            live_memory_path=live_memory_path,
            tasks=tasks,
            dry_run=dry_run,
            clock=clock,
            tokens_used=backend.tokens_used(),
        )

    gate_reason = _checkable_gate_reason(cfg, backend_name, tasks)
    if gate_reason:
        report.gate_action = "skip"
        report.no_edits_reason = gate_reason
        report.pre_gate_status = gate_reason
        report.notes.append(
            "skipped before replay/consolidation: "
            f"{report.n_checkable_tasks} checkable tasks and "
            f"{report.n_checkable_val_tasks} checkable val tasks; requires "
            f"{int(cfg.get('min_checkable_tasks', 3) or 0)} total and "
            f"{int(cfg.get('min_checkable_val_tasks', 2) or 0)} val"
        )
        report.notes.append("use a reviewed --tasks-file with concrete rule/rubric/exact checks")
        return _finish_without_consolidation(
            cfg,
            report=report,
            state=state,
            project=project,
            started=started,
            live_skill_path=live_skill_path,
            live_memory_path=live_memory_path,
            tasks=tasks,
            dry_run=dry_run,
            clock=clock,
            tokens_used=backend.tokens_used(),
        )

    # ── 3+4. replay + consolidate (gate), with opt-in dream + recall ──────
    # recall pulls similar past tasks from the persisted archive; dream_rollouts
    # / dream_factor enrich the training signal. With the defaults (recall_k=0,
    # dream_rollouts=1, dream_factor=0) this is exactly the prior single-shot
    # consolidate — behavior is unchanged unless the user opts in.
    _progress(cfg, "consolidate start")
    recall_k = int(cfg.get("recall_k", 0) or 0)
    history_tasks = []
    if recall_k > 0:
        history_tasks = [TaskRecord.from_dict(d) for d in state.task_archive()]
    result = dream_consolidate(
        backend, tasks, skill, memory,
        history_tasks=history_tasks,
        recall_k=recall_k,
        dream_rollouts=int(cfg.get("dream_rollouts", 1) or 1),
        dream_factor=int(cfg.get("dream_factor", 0) or 0),
        edit_budget=cfg.get("edit_budget", 4),
        gate_metric=cfg.get("gate_metric", "mixed"),
        gate_mixed_weight=cfg.get("gate_mixed_weight", 0.5),
        gate_mode=cfg.get("gate_mode", "on"),
        evolve_skill=cfg.get("evolve_skill", True),
        evolve_memory=cfg.get("evolve_memory", True),
        night=night,
    )
    # archive tonight's real (non-dream) tasks so future nights can recall them
    state.add_to_archive([t.to_dict() for t in tasks if t.origin != "dream"])
    _progress(
        cfg,
        f"consolidate done: gate={result.gate_action} accepted={result.accepted} "
        f"edits={len(result.applied_edits)} rejected={len(result.rejected_edits)}",
    )

    report.n_replayed = len(tasks)
    report.baseline_score = result.baseline_score
    report.candidate_score = result.candidate_score
    report.accepted = result.accepted
    report.gate_action = result.gate_action
    report.no_edits_reason = getattr(result, "no_edits_reason", "")
    report.edits = result.applied_edits
    report.rejected_edits = result.rejected_edits
    report.tokens_used = backend.tokens_used()
    report.ended_at = _now_iso(clock)

    # ── 5. stage (unless dry-run) ────────────────────────────────────────
    staging_dir = ""
    adopted = False
    adopted_paths: List[str] = []
    if not dry_run:
        _progress(cfg, "staging start")
        report_md = _render_report_md(report, cfg)
        proposed_skill = result.new_skill if (cfg.get("evolve_skill") and result.accepted) else None
        proposed_memory = result.new_memory if (cfg.get("evolve_memory") and result.accepted) else None
        staging_dir = write_staging(
            project,
            report=report,
            proposed_skill=proposed_skill,
            proposed_memory=proposed_memory,
            live_skill_path=live_skill_path,
            live_memory_path=live_memory_path,
            report_md=report_md,
        )
        state.set_last_harvest(project, started)
        state.record_night({
            "night": night, "accepted": result.accepted,
            "baseline": result.baseline_score, "candidate": result.candidate_score,
            "n_tasks": len(tasks),
            "n_checkable_tasks": report.n_checkable_tasks,
            "n_checkable_val_tasks": report.n_checkable_val_tasks,
            "miner_mode": report.miner_mode,
            "staging": staging_dir,
        })
        # ── 6. adopt (opt-in) ────────────────────────────────────────────
        if cfg.get("auto_adopt") and result.accepted:
            adopted_paths = adopt_staging(staging_dir)
            adopted = bool(adopted_paths)
        state.save()

    return CycleOutcome(report, staging_dir, adopted, adopted_paths)
