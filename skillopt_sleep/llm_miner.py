"""LLM-backed mining of replay-checkable tasks from session batches."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List, Optional

from skillopt_sleep.backend import Backend, _extract_json
from skillopt_sleep.types import SessionDigest, TaskRecord


_MINER_PROMPT = """You are mining a batch of past AI-assistant sessions for reusable tasks
worth optimizing a skill for. Return at most __MAX_TASKS__ tasks across the batch.

Prefer tasks that recur across sessions, received user corrections, or contain explicit
constraints that a general rule would help satisfy next time. A useful one-session task
is allowed when it exposes a reusable correction or convention. Skip truly one-off or
purely exploratory requests and anything that cannot be graded concretely.

For each task return:
  - "intent": a generalized reusable request with one-off details removed
  - "source_session_ids": session IDs supporting the task
  - "checks": programmatic checks, each using one supported operation:
      {"op":"section_present","arg":"<heading>"}
      {"op":"regex","arg":"<python regex>"}
      {"op":"contains","arg":"<required substring>"}
      {"op":"max_chars","arg":<positive integer>}
      {"op":"min_chars","arg":<positive integer>}
  - "rubric": one sentence describing success when exact checks are inappropriate
  - "satisfied": whether the prior result appeared satisfactory

Only include checks that every good future answer must satisfy. Return ONLY a JSON array,
possibly empty, with no prose or markdown fences.

# Sessions
__SESSIONS__
"""

_PROMPT_CHAR_BUDGET = 48_000


def _digest_block(digest: SessionDigest, char_budget: int) -> str:
    prompts = "\n".join(f"- {prompt}" for prompt in digest.user_prompts[:6]) or "- (none)"
    final = digest.assistant_finals[-1] if digest.assistant_finals else "(none)"
    block = (
        f"## session_id: {digest.session_id}\n"
        f"project: {digest.project or '(unknown)'}\n"
        f"user prompts:\n{prompts}\n"
        f"assistant final: {final}\n"
        f"feedback: {', '.join(digest.feedback_signals[:6]) or '(none)'}"
    )
    return block[:char_budget]


def _digests_to_prompt(digests: List[SessionDigest], max_tasks: int) -> str:
    usable = [digest for digest in digests if digest.user_prompts]
    per_session = max(240, min(1800, _PROMPT_CHAR_BUDGET // max(1, len(usable))))
    sessions = "\n\n".join(
        _digest_block(digest, per_session) for digest in usable
    )[:_PROMPT_CHAR_BUDGET]
    return (
        _MINER_PROMPT
        .replace("__MAX_TASKS__", str(max_tasks))
        .replace("__SESSIONS__", sessions or "(none)")
    )


def _clean_checks(checks: Any) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for check in checks if isinstance(checks, list) else []:
        if not isinstance(check, dict):
            continue
        op = check.get("op")
        arg = check.get("arg")
        if op in {"max_chars", "min_chars"}:
            if isinstance(arg, int) and not isinstance(arg, bool) and arg > 0:
                cleaned.append({"op": op, "arg": arg})
            continue
        if op not in {"section_present", "regex", "contains"}:
            continue
        if not isinstance(arg, str) or not arg.strip():
            continue
        if op == "regex":
            try:
                re.compile(arg)
            except re.error:
                continue
        cleaned.append({"op": op, "arg": arg.strip()})
    return cleaned


def _mk_task(
    digests_by_id: Dict[str, SessionDigest],
    fallback: SessionDigest,
    obj: Dict[str, Any],
) -> TaskRecord | None:
    intent = str(obj.get("intent", "")).strip()
    if len(intent) < 8:
        return None
    source_ids = [
        str(session_id)
        for session_id in (obj.get("source_session_ids") or [])
        if str(session_id) in digests_by_id
    ]
    if not source_ids:
        source_ids = [fallback.session_id]
    source = digests_by_id.get(source_ids[0], fallback)
    checks = _clean_checks(obj.get("checks"))
    rubric = str(obj.get("rubric", "")).strip()
    common = {
        "id": "llm_" + hashlib.sha256(intent.encode("utf-8")).hexdigest()[:12],
        "project": source.project,
        "intent": intent,
        "outcome": "success" if bool(obj.get("satisfied", False)) else "fail",
        "tags": ["mined:llm"],
        "source_sessions": source_ids,
    }
    if checks:
        return TaskRecord(
            **common,
            reference_kind="rule",
            judge={"kind": "rule", "checks": checks},
        )
    if rubric:
        return TaskRecord(**common, reference_kind="rubric", reference=rubric)
    return None


def make_llm_miner(
    backend: Backend,
    *,
    max_sessions: int = 20,
    max_tasks: int = 40,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Callable[[List[SessionDigest]], List[TaskRecord]]:
    """Return a miner that gives one bounded model call the full session batch."""

    def _miner(digests: List[SessionDigest]) -> List[TaskRecord]:
        selected = [digest for digest in digests[:max_sessions] if digest.user_prompts]
        if diagnostics is not None:
            diagnostics.setdefault("llm_parse_failures", 0)
            diagnostics.setdefault("llm_empty_responses", 0)
            diagnostics.setdefault("llm_backend_errors", 0)
            diagnostics.setdefault("llm_uncheckable_candidates", 0)
            diagnostics["llm_miner_failed"] = False
            diagnostics["sessions_passed_to_miner"] = len(selected)
            diagnostics["llm_sessions_processed"] = len(selected)
        if not selected:
            return []

        prompt = _digests_to_prompt(selected, max_tasks)
        parsed = None
        for attempt in range(2):
            try:
                suffix = "" if attempt == 0 else "\n\nRetry: return only a valid JSON array."
                raw = backend._call(  # type: ignore[attr-defined]
                    prompt + suffix,
                    max_tokens=max(800, min(4096, max_tasks * 300)),
                )
            except Exception as exc:
                if diagnostics is not None:
                    diagnostics["llm_backend_errors"] += 1
                    diagnostics["llm_miner_error"] = f"{type(exc).__name__}: {exc}"
                    diagnostics["llm_miner_failed"] = True
                break
            if not str(raw or "").strip():
                if diagnostics is not None:
                    diagnostics["llm_empty_responses"] += 1
                continue
            parsed = _extract_json(raw, "array")
            if isinstance(parsed, list):
                break
            if diagnostics is not None:
                diagnostics["llm_parse_failures"] += 1

        if not isinstance(parsed, list):
            if diagnostics is not None:
                diagnostics["llm_miner_failed"] = True
            return []
        if not parsed and diagnostics is not None:
            diagnostics["llm_empty_responses"] += 1

        by_id = {digest.session_id: digest for digest in selected}
        tasks: List[TaskRecord] = []
        for obj in parsed[:max_tasks]:
            task = _mk_task(by_id, selected[0], obj) if isinstance(obj, dict) else None
            if task is not None:
                tasks.append(task)
            elif diagnostics is not None:
                diagnostics["llm_uncheckable_candidates"] += 1
        if diagnostics is not None:
            diagnostics["llm_tasks_returned"] = len(tasks)
            if parsed and not tasks:
                diagnostics["llm_miner_failed"] = True
        return tasks

    return _miner
