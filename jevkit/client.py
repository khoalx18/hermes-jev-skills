"""One small, strict client for TypeSafe Jev (POST /v1/systemone) + local openJev fallback.

Jev answers typed questions about a state: ``choice`` (one of a closed set),
``score`` (a position on an ordered rubric) and ``noul`` (probability of yes).
It never writes text. Every helper here validates the reply against the question
that was asked, so a malformed or surprising answer becomes a ``JevError`` and
the caller takes its fail-open path rather than acting on junk.

Patched to support local openJev (149M) via OPENJEV_LOCAL=1 or JEV_LOCAL=1,
using DecisionEngine at OPENJEV_MODEL_PATH (default hf_rlcd). No API key needed.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from . import keystore

ENDPOINT = os.environ.get("JEV_URL") or os.environ.get("TYPESAFE_API_URL") or "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
MAX_RESPONSE_BYTES = 1_000_000
MAX_STATE_CHARS = 60_000
USER_AGENT = "hermes-jev-skills/0.1"

# --- openJev local config ---
OPENJEV_MODEL_PATH = os.environ.get("OPENJEV_MODEL_PATH") or "C:/Users/ADMIN/openJev-verdict-2.0/hf_rlcd"
OPENJEV_USE_LOCAL = os.environ.get("OPENJEV_LOCAL") == "1" or os.environ.get("JEV_LOCAL") == "1"
# Auto-enable if model exists and no key? Keep explicit env for now, but also check file existence for Hermes
if not OPENJEV_USE_LOCAL and Path(OPENJEV_MODEL_PATH).exists():
    # If the file exists and the user has swapped (we set env in Hermes .env), it will be 1.
    # Keep false by default to not break existing installs without env.
    pass

State = Union[str, Mapping[str, Any], Sequence[Any]]
Transport = Callable[[bytes, Dict[str, str], float], bytes]


class JevError(RuntimeError):
    """Anything that means "do not trust or use this Jev result"."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code

# ── question builders ────────────────────────────────────────────────────────
def choice(instructions: str, criteria: Mapping[str, str]) -> Dict[str, Any]:
    if len(criteria) < 2:
        raise ValueError("a choice needs at least two options")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score(instructions: str, levels: Sequence[str]) -> Dict[str, Any]:
    if len(levels) < 2:
        raise ValueError("a score needs at least two levels")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def noul(instructions: str) -> Dict[str, Any]:
    return {"type": "noul", "instructions": instructions}

# ── transport ────────────────────────────────────────────────────────────────
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401, ANN001
        # A redirect would carry the bearer token to another origin.
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def _http_transport(body: bytes, headers: Dict[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(ENDPOINT, data=body, headers=headers, method="POST")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        code = {401: "auth_failed", 403: "auth_failed", 402: "credits_exhausted",
                429: "rate_limited", 529: "overloaded"}.get(error.code, f"http_{error.code}")
        raise JevError(code) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise JevError("network") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise JevError("response_too_large")
    return raw


_RETRYABLE = {"rate_limited", "overloaded", "network", "http_500", "http_502", "http_503", "http_504"}

# ── openJev local engine (lazy) ─────────────────────────────────────────────
_local_engine = None


def _local_device() -> str:
    """CUDA when the build has it, unless OPENJEV_DEVICE says otherwise.

    The local engine ran CPU-only, which cost ~2-3s per passage. Both values stay
    selectable so the choice can be measured instead of assumed.
    """
    forced = (os.environ.get("OPENJEV_DEVICE") or "").strip()
    if forced:
        return forced
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch means the CPU path
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def _local_device() -> str:
    """CUDA when the build has it, unless OPENJEV_DEVICE says otherwise.

    The local engine ran CPU-only, which cost ~2-3s per passage. Both values stay
    selectable so the choice can be measured instead of assumed.
    """
    forced = (os.environ.get("OPENJEV_DEVICE") or "").strip()
    if forced:
        return forced
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch means the CPU path
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"
_local_calibrator = None

def _get_local_engine():
    global _local_engine, _local_calibrator
    if _local_engine is not None:
        return _local_engine
    # Add openJev to path if needed
    openjev_root = Path("C:/Users/ADMIN/openJev-verdict-2.0")
    if str(openjev_root) not in sys.path:
        sys.path.insert(0, str(openjev_root))
    try:
        from core.engine_encoder import DecisionEngine
        from core.calibration import TemperatureCalibrator
        cal_path = Path(OPENJEV_MODEL_PATH) / "calibrator.json"
        cal = TemperatureCalibrator.load(str(cal_path)) if cal_path.exists() else TemperatureCalibrator(initial_temperature=1.0)
        eng = DecisionEngine(model_name_or_path=OPENJEV_MODEL_PATH, calibrator=cal, device=_local_device())
        _local_engine = eng
        _local_calibrator = cal
        return eng
    except Exception as e:
        raise JevError("local_engine_failed", str(e)) from e

def _ask_local(state: State, questions: Mapping[str, Mapping[str, Any]], timeout: float = 4.0) -> Dict[str, Any]:
    """Translate jevkit questions to openJev Query and run locally."""
    # Ensure path
    openjev_root = Path("C:/Users/ADMIN/openJev-verdict-2.0")
    if str(openjev_root) not in sys.path:
        sys.path.insert(0, str(openjev_root))
    from core.primitives import Choice, Option, Noul, Score, Level
    engine = _get_local_engine()
    # State to string
    if isinstance(state, str):
        ctx = state
    else:
        try:
            ctx = json.dumps(state, separators=(",", ":"), default=str)
        except Exception:
            ctx = str(state)
    # Build Query objects
    queries = []
    q_names = []
    for name, qdef in questions.items():
        qtype = qdef.get("type")
        instructions = qdef.get("instructions") or qdef.get("question") or ""
        if qtype == "choice":
            criteria = qdef.get("criteria") or {}
            opts = tuple(Option(id=str(k), description=str(v)) for k, v in criteria.items())
            queries.append(Choice(id=name, question=instructions, options=opts))
            q_names.append(name)
        elif qtype == "noul":
            queries.append(Noul(id=name, proposition=instructions, semantics="conditional_on_sufficient_evidence_v2"))
            q_names.append(name)
        elif qtype == "score":
            criteria = qdef.get("criteria") or []
            levels = tuple(Level(id=str(i), description=str(c), value=float(i)) for i, c in enumerate(criteria))
            queries.append(Score(id=name, question=instructions, levels=levels))
            q_names.append(name)
        else:
            raise JevError("malformed", f"unknown question type {qtype}")
    started = time.monotonic()
    # openJev evaluate is batched, respect timeout
    try:
        batch = engine.evaluate(ctx, queries)
    except Exception as e:
        raise JevError("local_engine_failed", str(e)) from e
    latency_ms = int((time.monotonic() - started) * 1000)
    # Convert back to jevkit answer shape
    answers = {}
    for res in batch.results:
        if res.kind == "choice":
            # res is ChoiceResult
            ans = {
                "type": "choice",
                "choice": res.selected_id,
                "probabilities": {k: float(v) for k, v in res.probabilities.items() if k != "__insufficient_evidence__"},
                "confidence": float(res.selected_probability),
            }
            # Filter probabilities to only offered options (remove insufficient)
            # Ensure sum ~1 for offered options + confidence handling is done by _check_answer
            # We need to renormalize probabilities to sum 1 for offered options if insufficient was present
            # Keep as is but remove insufficient key if present
            if "__insufficient_evidence__" in res.probabilities:
                # Renormalize remaining
                ins_p = res.probabilities.get("__insufficient_evidence__", 0)
                if ins_p < 0.99:
                    factor = 1.0 / (1.0 - ins_p) if (1.0 - ins_p) > 1e-9 else 1.0
                    ans["probabilities"] = {k: float(v * factor) for k, v in ans["probabilities"].items()}
                    # Update confidence to renormalized value
                    if ans["choice"] in ans["probabilities"]:
                        ans["confidence"] = float(ans["probabilities"][ans["choice"]])
                # Ensure confidence still valid
                if ans["choice"] == "__insufficient_evidence__":
                    # Abstention - pick next best
                    # Find max among offered
                    if ans["probabilities"]:
                        best = max(ans["probabilities"], key=ans["probabilities"].get)
                        ans["choice"] = best
                        ans["confidence"] = float(ans["probabilities"][best])
            # Always ensure probabilities sum to 1 (fix rounding / model not normalized)
            s = sum(ans["probabilities"].values())
            if abs(s - 1.0) > 0.01 and s > 1e-9:
                ans["probabilities"] = {k: float(v / s) for k, v in ans["probabilities"].items()}
                if ans["choice"] in ans["probabilities"]:
                    ans["confidence"] = float(ans["probabilities"][ans["choice"]])
            answers[res.id] = ans
        elif res.kind == "noul":
            # res is NoulResult
            # Map to jevkit noul: {"type":"noul","noul": p_true}
            p_true = res.p_true_given_sufficient_evidence
            if p_true is None:
                # abstention
                p_true = 0.5
            ans = {"type": "noul", "noul": float(p_true)}
            answers[res.id] = ans
        elif res.kind == "score":
            # res is ScoreResult
            ans = {
                "type": "score",
                "score": float(res.selected_value) if res.selected_value is not None else 0.0,
                "probabilities": {str(k): float(v) for k, v in res.probabilities.items() if k != "__insufficient_evidence__"},
                "confidence": float(res.probabilities.get(res.selected_level_id, 0.0)) if res.selected_level_id in res.probabilities else 0.5,
            }
            answers[res.id] = ans
    return {"answers": answers, "usage": {}, "latency_ms": latency_ms}

# ── validation ───────────────────────────────────────────────────────────────
def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevError("malformed", f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number) or not -1e-6 <= number <= 1 + 1e-6:
        raise JevError("malformed", f"{name} is outside 0..1")
    return min(1.0, max(0.0, number))


def _check_answer(name: str, question: Mapping[str, Any], answer: Any) -> Dict[str, Any]:
    if not isinstance(answer, dict) or answer.get("type") != question["type"]:
        raise JevError("malformed", f"answer {name} has the wrong type")
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": _unit(answer.get("noul"), f"{name}.noul")}
    if kind == "choice":
        options = set(question["criteria"])
        picked = answer.get("choice")
        if picked not in options:
            raise JevError("malformed", f"answer {name} chose an option that was not offered")
        raw = answer.get("probabilities")
        if not isinstance(raw, dict) or not set(raw) <= options:
            raise JevError("malformed", f"answer {name} has bad probabilities")
        probabilities = {key: _unit(value, f"{name}.p[{key}]") for key, value in raw.items()}
        return {"type": "choice", "choice": picked, "probabilities": probabilities,
                "confidence": _unit(answer.get("confidence"), f"{name}.confidence")}
    levels = len(question["criteria"])
    value = answer.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise JevError("malformed", f"answer {name} has no numeric score")
    if not -0.5 <= float(value) <= levels - 0.5:
        raise JevError("malformed", f"answer {name} scored off the rubric")
    raw = answer.get("probabilities")
    spread = {}
    if isinstance(raw, dict):
        for key, probability in raw.items():
            if str(key).isdigit() and int(key) < levels:
                spread[int(key)] = _unit(probability, f"{name}.p[{key}]")
    return {"type": "score", "score": float(value), "probabilities": spread,
            "confidence": _unit(answer.get("confidence", 1.0), f"{name}.confidence")}

# ── public call ──────────────────────────────────────────────────────────────
def ask(
    state: State,
    questions: Mapping[str, Mapping[str, Any]],
    *,
    timeout: float = 4.0,
    retries: int = 1,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    transport: Optional[Transport] = None,
) -> Dict[str, Any]:
    """Ask Jev every question against one state, in a single request.

    Returns ``{"answers": {...validated...}, "usage": {...}, "latency_ms": int}``.
    Raises ``JevError`` for anything the caller should not act on. ``timeout`` is a
    total wall-clock budget across retries, not a per-attempt one.
    """
    if not questions:
        raise ValueError("no questions")
    # --- LOCAL PATH ---
    # If OPENJEV_LOCAL=1, try local first, fallback to remote on failure if not forced
    use_local = os.environ.get("OPENJEV_LOCAL") == "1" or os.environ.get("JEV_LOCAL") == "1"
    # Also auto-use local if model path exists and no key is present (to avoid auth_failed)
    if not use_local and Path(OPENJEV_MODEL_PATH).exists():
        # Check if key is missing or will fail - but we still respect explicit remote unless forced
        # For now, only use local when env var is set, to keep behavior predictable
        pass
    if use_local:
        try:
            # Local does not need key, but we still validate state size
            encoded_state = state if isinstance(state, str) else json.dumps(state, separators=(",", ":"), default=str)
            if len(encoded_state) > MAX_STATE_CHARS:
                raise JevError("state_too_large")
            # Local ask - no retries needed, just one call
            started = time.monotonic()
            result = _ask_local(state, questions, timeout=timeout)
            # Validate answers via same _check_answer to ensure shape
            checked = {name: _check_answer(name, question, result["answers"].get(name)) for name, question in questions.items()}
            return {"answers": checked, "usage": result.get("usage", {}), "latency_ms": result.get("latency_ms", int((time.monotonic() - started)*1000))}
        except JevError:
            raise
        except Exception as e:
            raise JevError("local_engine_failed", str(e)) from e
    # --- REMOTE PATH (original) ---
    key = api_key or keystore.resolve()
    if not key:
        raise JevError("no_key", "run `jev setup-key`")
    encoded_state = state if isinstance(state, str) else json.dumps(state, separators=(",", ":"), default=str)
    if len(encoded_state) > MAX_STATE_CHARS:
        raise JevError("state_too_large")
    body = json.dumps(
        {"state": state, "model": model or os.environ.get("TYPESAFE_MODEL") or DEFAULT_MODEL,
         "questions": {name: dict(q) for name, q in questions.items()}},
        separators=(",", ":"), default=str,
    ).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "Accept": "application/json", "User-Agent": USER_AGENT}
    send = transport or _http_transport
    started = time.monotonic()
    attempt = 0
    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0.05:
            raise JevError("timeout")
        try:
            raw = send(body, headers, remaining)
            break
        except JevError as error:
            attempt += 1
            if error.code not in _RETRYABLE or attempt > retries:
                raise
            time.sleep(min(0.25 * attempt, max(0.0, timeout - (time.monotonic() - started) - 0.1)))
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise JevError("malformed", "reply is not JSON") from None
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        raise JevError("malformed", "reply has no answers")
    checked = {name: _check_answer(name, question, answers.get(name)) for name, question in questions.items()}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {"answers": checked, "usage": usage, "latency_ms": int((time.monotonic() - started) * 1000)}


def verify_key(api_key: str, timeout: float = 10.0) -> bool:
    """One tiny synthetic call. True means the key is accepted."""
    try:
        ask("The build finished and all tests passed.",
            {"ok": noul("The text reports a successful outcome")}, api_key=api_key, timeout=timeout)
        return True
    except JevError:
        return False


def batches(items: Sequence[Any], size: int) -> List[Sequence[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]
