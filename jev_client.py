"""JEV 接入层 —— 把 TypeSafe 的 Jev 当"结构化裁判"用。

设计原则（与 cloud_provider.py 一致）：
1. **默认关闭**。没有 key、网络不通、地区封锁，都不能拖垮主流程。
2. **失败即降级**。任何异常都返回 {ok: False, ...}，调用方自己决定要不要用。
3. **不写 key**。key 只从参数或环境变量取，绝不落到文件、日志、返回值里。
4. **只用真实响应**。没拿到 answers 就是没拿到，不编。

用法：
    from jev_client import ask_jev, jev_available
    if jev_available(key):
        r = ask_jev(state="...", questions={...}, api_key=key)
        if r["ok"]:
            print(r["answers"])
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
MAX_QUESTIONS = 8
MAX_STATE_CHARS = 8000


class JevError(Exception):
    """可预期的失败（认证、地区封锁、超时），调用方按降级处理。"""


def _key_from_env() -> str | None:
    for name in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
        v = os.environ.get(name)
        if v:
            return v.strip()
    return None


def jev_available(api_key: str | None = None) -> bool:
    """有没有可用的 key。不代表 key 一定有效——那是 ask_jev 的事。"""
    return bool((api_key or _key_from_env()))


def _validate(questions: dict) -> None:
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions 必须是非空 dict")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"一次最多 {MAX_QUESTIONS} 个问题")
    allowed = {"noul", "choice", "score"}
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise ValueError(f"{qid}: 必须是对象")
        if q.get("type") not in allowed:
            raise ValueError(f"{qid}: type 必须是 {sorted(allowed)}")
        if not str(q.get("instructions", "")).strip():
            raise ValueError(f"{qid}: 缺少 instructions")
        if q["type"] in ("choice", "score") and not q.get("criteria"):
            raise ValueError(f"{qid}: {q['type']} 需要 criteria")


def ask_jev(
    state: str,
    questions: dict,
    api_key: str | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    timeout: int = 30,
) -> dict:
    """问 Jev 一组结构化问题。

    返回：
        {"ok": True,  "answers": {...}, "model": "jev-x.y.z", "raw_status": 200}
        {"ok": False, "error": "auth|region|timeout|http_NN|bad_response", "detail": "..."}
    """
    key = api_key or _key_from_env()
    if not key:
        return {"ok": False, "error": "no_key", "detail": "没有 TYPESAFE_API_KEY / JEV_API_KEY"}

    if not str(state or "").strip():
        return {"ok": False, "error": "no_state", "detail": "state 为空"}
    if len(state) > MAX_STATE_CHARS:
        state = state[:MAX_STATE_CHARS]

    try:
        _validate(questions)
    except ValueError as exc:
        return {"ok": False, "error": "bad_request", "detail": str(exc)}

    body = json.dumps({"state": state, "model": model, "questions": questions}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "User-Agent": "EvoResearch/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            payload = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        payload = exc.read().decode("utf-8", "replace") if exc.fp else ""
        # 401/403 = key 无效或没给；451 = 地区封锁
        kind = {401: "auth", 403: "auth", 451: "region"}.get(status, f"http_{status}")
        return {"ok": False, "error": kind, "detail": payload[:300], "raw_status": status}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "error": "timeout", "detail": str(exc)[:300]}
    except JevError as exc:
        return {"ok": False, "error": "transport", "detail": str(exc)[:300]}

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {"ok": False, "error": "bad_response", "detail": payload[:300], "raw_status": status}

    if not isinstance(data, dict) or "answers" not in data:
        return {"ok": False, "error": "bad_response", "detail": payload[:300], "raw_status": status}

    return {
        "ok": True,
        "answers": data["answers"],
        "model": data.get("model", ""),
        "raw_status": status,
    }


# ---------------- 业务封装：三个固定问题 ----------------

REVIEW_QUESTIONS = {
    "is_bug": {
        "type": "noul",
        "instructions": "这段实验结论指向的是一个真实的实现问题（不是表达或格式问题）",
    },
    "severity": {
        "type": "score",
        "instructions": "这个问题有多严重",
        "criteria": ["可以忽略", "值得记录", "必须修复", "会毁掉结论"],
    },
    "category": {
        "type": "choice",
        "instructions": "这属于哪一类问题",
        "criteria": {
            "leakage": "数据泄漏",
            "metric": "度量口径错误",
            "reproducibility": "不可复现",
            "budget": "预算越界",
            "other": "其他",
        },
    },
}


def review_finding(state: str, api_key: str | None = None) -> dict:
    """给一条实验发现做结构化裁定。失败就返回 ok=False，调用方走规则兜底。"""
    return ask_jev(state=state, questions=REVIEW_QUESTIONS, api_key=api_key)


if __name__ == "__main__":
    import sys

    print("jev_available:", jev_available())
    if len(sys.argv) > 1:
        r = ask_jev(state=sys.argv[1], questions=REVIEW_QUESTIONS)
        print(json.dumps(r, ensure_ascii=False, indent=2))
