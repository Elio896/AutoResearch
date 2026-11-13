# -*- coding: utf-8 -*-
"""
validate_trace.py —— 研究轨迹的机器验收
=======================================
把「九项证据」从口号变成可执行检查。用法：

    python validate_trace.py runs/demo_readmission
    python validate_trace.py runs/demo_readmission --strict   # WARN 也算失败

退出码：0 通过 / 1 有 FAIL / 2 只有 WARN（--strict 下为 1）。

检查分两层：
  L1 结构：按 round_record.schema.json 逐字段校验（内置迷你校验器，零依赖）
  L2 语义：哈希链、结果自洽、预算算术、回滚指向、复现三元组
L2 才是「这轮能不能算证据」的关键；L1 只保证字段没写错。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "round_record.schema.json"
LEVELS = {"FAIL": 2, "WARN": 1, "PASS": 0}


# ------------------------------------------------------------------ 迷你 schema 校验
def _type_ok(value, t) -> bool:
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "null":
        return value is None
    return True


def check_schema(node, schema, path, errors):
    """支持本项目 schema 用到的关键字子集。"""
    if "type" in schema:
        ts = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(node, t) for t in ts):
            errors.append(f"{path}: 类型应为 {ts}，实际 {type(node).__name__}")
            return

    if "const" in schema and node != schema["const"]:
        errors.append(f"{path}: 应为常量 {schema['const']!r}")
    if "enum" in schema and node not in schema["enum"]:
        errors.append(f"{path}: 值 {node!r} 不在枚举 {schema['enum']}")

    if isinstance(node, str):
        if "pattern" in schema and not re.search(schema["pattern"], node):
            errors.append(f"{path}: 不匹配 {schema['pattern']}")
        if "minLength" in schema and len(node) < schema["minLength"]:
            errors.append(f"{path}: 长度 {len(node)} < {schema['minLength']}")

    if isinstance(node, (int, float)) and not isinstance(node, bool):
        if "minimum" in schema and node < schema["minimum"]:
            errors.append(f"{path}: {node} < minimum {schema['minimum']}")
        if "maximum" in schema and node > schema["maximum"]:
            errors.append(f"{path}: {node} > maximum {schema['maximum']}")

    if isinstance(node, list):
        if "minItems" in schema and len(node) < schema["minItems"]:
            errors.append(f"{path}: 元素数 {len(node)} < {schema['minItems']}")
        if "maxItems" in schema and len(node) > schema["maxItems"]:
            errors.append(f"{path}: 元素数 {len(node)} > {schema['maxItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, v in enumerate(node):
                check_schema(v, item_schema, f"{path}[{i}]", errors)

    if isinstance(node, dict):
        for key in schema.get("required", []):
            if key not in node:
                errors.append(f"{path}: 缺少必填字段 {key}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in node:
                if key not in props:
                    errors.append(f"{path}: 不允许的字段 {key}")
        for key, sub in props.items():
            if key in node:
                check_schema(node[key], sub, f"{path}.{key}", errors)


# ------------------------------------------------------------------ 哈希
def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def record_hash(rec: dict) -> str:
    body = json.loads(json.dumps(rec))          # 深拷贝
    body.get("chain", {}).pop("hash", None)
    return "sha256:" + hashlib.sha256(canon(body)).hexdigest()


# ------------------------------------------------------------------ 语义检查
def check_semantics(rec, prev, by_id, res):
    rid = rec.get("round_id", "?")
    rnd = rec.get("round", "?")

    # 3 哈希链
    ch = rec.get("chain", {})
    if ch.get("hash") != record_hash(rec):
        res.append(("FAIL", rnd, rid, "CHAIN_HASH", "记录的 self hash 与内容不符（被改过）"))
    expect_prev = prev.get("chain", {}).get("hash") if prev else None
    if ch.get("prev_hash") != expect_prev:
        res.append(("FAIL", rnd, rid, "CHAIN_BREAK",
                    f"prev_hash={ch.get('prev_hash')!r}，应为 {expect_prev!r}"))

    # 4 结果自洽
    r = rec.get("result", {})
    val, base, delta = r.get("value"), r.get("baseline_value"), r.get("delta")
    if None not in (val, base, delta):
        if abs((val - base) - delta) > 1e-6:
            res.append(("FAIL", rnd, rid, "DELTA_MATH",
                        f"delta={delta} 与 value-baseline={val - base:.4f} 不符"))
        if r.get("delta_significant") and abs(delta) < (r.get("mde") or 0):
            res.append(("FAIL", rnd, rid, "SIG_CLAIM", "声称显著，但 |delta| < mde"))
        pf = r.get("per_fold") or []
        if pf:
            mean = sum(pf) / len(pf)
            if abs(mean - val) > 0.02:
                res.append(("WARN", rnd, rid, "FOLD_MEAN",
                            f"per_fold 均值 {mean:.3f} 与 value {val} 差 > 0.02（口径不同请写清）"))
        ci = r.get("ci95") or []
        if len(ci) == 2 and not (ci[0] - 0.02 <= val <= ci[1] + 0.02):
            res.append(("FAIL", rnd, rid, "CI_COVER", f"value {val} 不在 ci95 {ci} 内"))

    # 5 预算
    b = rec.get("budget", {})
    cum, cap = b.get("cumulative", {}), b.get("cap", {})
    for k, capk in (("tokens", "tokens"), ("wall_s", "wall_s"), ("experiments", "experiments")):
        if k in cum and capk in cap and cum[k] > cap[capk]:
            res.append(("FAIL", rnd, rid, "BUDGET_OVER",
                        f"{k} 累计 {cum[k]} 超上限 {cap[capk]}"))
    tok = b.get("tokens_in", 0) + b.get("tokens_out", 0)
    rec_tok = b.get("reconciled_with", {}).get("tokens")
    if rec_tok is not None and abs(rec_tok - cum.get("tokens", 0)) > max(1, 0.01 * max(1, cum.get("tokens", 1))):
        res.append(("FAIL", rnd, rid, "BUDGET_RECON",
                    f"自记 {cum.get('tokens')} vs 网关 {rec_tok}，误差超 1%"))
    if tok == 0 and rec.get("exec", {}).get("stage") == "experiment_only":
        pass                                    # 只跑执行器，0 token 合理
    elif tok == 0:
        res.append(("WARN", rnd, rid, "TOKEN_ZERO", "声明用了 LLM 但 token 为 0"))

    # 6 verdict 一致
    if rec.get("status") != rec.get("conclusion", {}).get("verdict"):
        res.append(("FAIL", rnd, rid, "VERDICT_MISMATCH",
                    f"status={rec.get('status')} ≠ conclusion.verdict"))

    # 7 回滚指向
    v = rec.get("validation", {})
    if v.get("reverted"):
        target = v.get("restored_from")
        if not target:
            res.append(("FAIL", rnd, rid, "REVERT_NO_TARGET", "reverted=true 但没写 restored_from"))
        elif target not in by_id:
            res.append(("FAIL", rnd, rid, "REVERT_BAD_TARGET", f"restored_from 指向不存在的轮次 {target}"))
        if rec.get("status") != "reverted":
            res.append(("FAIL", rnd, rid, "REVERT_STATUS", "reverted=true 但 status 不是 reverted"))

    # 8 假设可证伪
    h = rec.get("hypothesis", {})
    if h.get("expected_direction") == "up" and (delta or 0) < 0 and rec.get("status") == "kept":
        res.append(("WARN", rnd, rid, "HYP_DIRECTION", "预期上升但结果下降，仍被采纳"))
    if len(h.get("falsifier", "")) < 10:
        res.append(("FAIL", rnd, rid, "FALSIFIER_WEAK", "falsifier 太短，不可判真假"))

    # 9 复现三元组
    sp = rec.get("input", {}).get("split", {})
    if not sp.get("split_hash") or sp.get("seed") is None or not rec.get("exec", {}).get("env_hash"):
        res.append(("FAIL", rnd, rid, "REPRO_TRIPLET", "缺 split_hash / seed / env_hash 之一"))


def check_evidence(rec, res):
    """九项证据是否齐全且非空。"""
    rid, rnd = rec.get("round_id", "?"), rec.get("round", "?")
    items = [
        ("01 身份", bool(rec.get("round_id")) and rec.get("chain", {}).get("prev_hash") is not None or rec.get("round") == 0),
        ("02 输入", bool(rec.get("input", {}).get("sha256")) and rec.get("input", {}).get("rows")),
        ("03 假设", bool(rec.get("hypothesis", {}).get("text")) and bool(rec.get("hypothesis", {}).get("falsifier"))),
        ("04 diff", (rec.get("intervention", {}).get("kind") == "baseline"
                     or len(rec.get("intervention", {}).get("diff") or []) > 0)),
        ("05 执行", bool(rec.get("exec", {}).get("command")) and rec.get("exec", {}).get("exit_code") is not None),
        ("06 结果", bool(rec.get("result", {}).get("per_fold")) and rec.get("result", {}).get("ci95") is not None),
        ("07 闸门", len(rec.get("validation", {}).get("gates") or []) > 0),
        ("08 预算", bool(rec.get("budget", {}).get("cap")) and bool(rec.get("budget", {}).get("reconciled_with"))),
        ("09 结论", bool(rec.get("conclusion", {}).get("reason")) and bool(rec.get("conclusion", {}).get("next_hypothesis"))),
    ]
    for name, ok in items:
        if not ok:
            res.append(("FAIL", rnd, rid, "EVIDENCE", f"证据 {name} 缺失或为空"))


# ------------------------------------------------------------------ 主流程
def main() -> int:
    try:                                        # Windows 控制台默认 GBK，避免特殊符号炸掉
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="校验一个 EvoResearch run 目录")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--strict", action="store_true", help="WARN 也算失败")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"找不到目录：{run_dir}")
        return 1

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    round_files = sorted((run_dir / "rounds").glob("*/round_record.json"))
    if not round_files:
        print("没有找到 rounds/*/round_record.json")
        return 1

    records, res = [], []
    for f in round_files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            res.append(("FAIL", "?", f.parent.name, "JSON", f"解析失败：{e}"))
            continue
        errs = []
        check_schema(rec, schema, "$", errs)
        for e in errs[:20]:
            res.append(("FAIL", rec.get("round", "?"), rec.get("round_id", f.parent.name),
                        "SCHEMA", e))
        check_evidence(rec, res)
        records.append(rec)

    records.sort(key=lambda r: r.get("round", 0))
    by_id = {r.get("round_id"): r for r in records}
    prev = None
    for rec in records:
        check_semantics(rec, prev, by_id, res)
        prev = rec

    # trace.jsonl 与记录一致
    trace_path = run_dir / "trace.jsonl"
    if trace_path.exists():
        lines = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) != len(records):
            res.append(("FAIL", "-", "-", "TRACE_LINES",
                        f"trace.jsonl 有 {len(lines)} 行，记录有 {len(records)} 个"))
        for line, rec in zip(lines, records):
            if line.get("hash") != rec.get("chain", {}).get("hash"):
                res.append(("FAIL", rec.get("round"), rec.get("round_id"), "TRACE_HASH",
                            "trace.jsonl 的 hash 与记录文件不一致"))
    else:
        res.append(("WARN", "-", "-", "NO_TRACE", "缺 trace.jsonl（哈希链只在单文件内校验）"))

    # 轮数
    if len(records) < 3:
        res.append(("WARN", "-", "-", "ROUNDS_LT3", f"只有 {len(records)} 轮，不足以体现演进"))

    # 输出
    res.sort(key=lambda x: (-LEVELS[x[0]], x[1] if isinstance(x[1], int) else 0))
    if args.json:
        print(json.dumps([{"level": a, "round": b, "round_id": c, "code": d, "msg": e}
                          for a, b, c, d, e in res], ensure_ascii=False, indent=2))
    else:
        if not res:
            print(f"[OK] {run_dir.name}: {len(records)} 轮全部通过，九项证据齐全，哈希链完整。")
        else:
            print(f"{'级别':<6}{'轮':<4}{'代码':<16}说明")
            print("-" * 78)
            for lvl, rnd, rid, code, msg in res:
                print(f"{lvl:<6}{str(rnd):<4}{code:<16}{msg}")
        fails = sum(1 for r in res if r[0] == "FAIL")
        warns = sum(1 for r in res if r[0] == "WARN")
        print(f"\n{len(records)} 轮 | FAIL {fails} | WARN {warns} | "
              f"{'不通过' if fails or (args.strict and warns) else '通过'}")

    if any(r[0] == "FAIL" for r in res):
        return 1
    if args.strict and any(r[0] == "WARN" for r in res):
        return 1
    if any(r[0] == "WARN" for r in res):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
