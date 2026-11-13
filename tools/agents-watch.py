#!/usr/bin/env python3
"""agents-watch.py - one timeline for every agent working on this project.

What it reads (all read-only):
  feed      .agent-log/events.jsonl                 Cline hooks write here automatically;
                                                     Codex and 鱼鱼 append lines themselves
  cline     ~/.cline/data/sessions/**/*.messages.json   Cline's own conversation log
  codex     ~/.codex/sessions/**/rollout-*.jsonl        Codex rollout transcripts (when present)
  project   file modifications under the project root
  status    liveness per agent: last log/last message/last write + running process

Usage
  python tools/agents-watch.py              # last 40 events + status + live processes
  python tools/agents-watch.py -n 100
  python tools/agents-watch.py --follow     # refresh every 3 s (Ctrl+C stops)
  python tools/agents-watch.py --json
  python tools/agents-watch.py --no-procs   # skip the process scan

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

HOME = Path.home()
PROJECT = Path(r"C:\workspace\EvoResearch")
FEED = PROJECT / ".agent-log" / "events.jsonl"
CLINE_SESSIONS = HOME / ".cline" / "data" / "sessions"
CODEX_SESSIONS = HOME / ".codex" / "sessions"
CODEX_LOGS = HOME / ".codex" / "logs_2.sqlite"

SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", ".agent-log", "artifacts", "node_modules"}


# ---------------------------------------------------------------- time helpers
def parse_ts(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 1000.0 if v > 1e11 else v
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{10,16}", s):
        v = float(s)
        return v / 1000.0 if v > 1e11 else v
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def stamp(ts: float | None) -> str:
    """HH:MM:SS today, MM-DD HH:MM:SS otherwise."""
    if not ts:
        return "--:--:--"
    try:
        dt = datetime.fromtimestamp(ts)
    except (OverflowError, OSError, ValueError):
        return "--:--:--"
    if dt.date() == date.today():
        return dt.strftime("%H:%M:%S")
    return dt.strftime("%m-%d %H:%M:%S")


def age(ts: float | None) -> str:
    if not ts:
        return "从无"
    d = time.time() - ts
    if d < 0:
        return "刚"
    if d < 60:
        return f"{int(d)}s前"
    if d < 3600:
        return f"{int(d / 60)}m前"
    if d < 86400:
        return f"{int(d / 3600)}h前"
    return f"{int(d / 86400)}d前"


def clip(text, n=160) -> str:
    if text is None:
        return ""
    s = re.sub(r"\s+", " ", str(text)).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def ev(ts, agent, kind, text):
    return {"ts": ts, "agent": agent, "kind": kind, "text": text}


# ------------------------------------------------------------------- feeders
def read_feed() -> list[dict]:
    out: list[dict] = []
    if not FEED.exists():
        return out
    try:
        with FEED.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                agent = o.get("agent") or "?"
                label = "cline" if agent == "cline" else agent
                out.append(ev(parse_ts(o.get("ts")), label, o.get("event") or "?", clip(o.get("summary"), 200)))
    except OSError:
        pass
    return out


def content_text(parts) -> str:
    if isinstance(parts, str):
        try:
            parts = json.loads(parts)
        except json.JSONDecodeError:
            return parts
    if not isinstance(parts, list):
        return str(parts or "")
    pieces = []
    for p in parts:
        if not isinstance(p, dict):
            pieces.append(str(p))
            continue
        t = p.get("type")
        if t == "text":
            pieces.append(p.get("text", ""))
        elif t == "thinking":
            pieces.append("[想] " + str(p.get("thinking", "")))
        elif t == "tool_use":
            name = p.get("name") or p.get("toolName") or "tool"
            pieces.append(f"[{name}] " + clip(json.dumps(p.get("input") or {}, ensure_ascii=False), 120))
        elif t == "tool_result":
            pieces.append("[结果] " + clip(p.get("content"), 120))
        else:
            pieces.append(f"[{t}]")
    return " ".join(x for x in pieces if x)


def read_cline(per_session: int = 8) -> tuple[list[dict], list[dict]]:
    events: list[dict] = []
    sessions: list[dict] = []
    if not CLINE_SESSIONS.exists():
        return events, sessions

    try:
        dirs = sorted(
            (d for d in CLINE_SESSIONS.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )[:3]
    except OSError:
        return events, sessions

    for d in dirs:
        meta_path = d / f"{d.name}.json"
        msgs_path = d / f"{d.name}.messages.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        last_msg_ts = None
        if msgs_path.exists():
            try:
                doc = json.loads(msgs_path.read_text(encoding="utf-8", errors="replace"))
                msgs = doc.get("messages") or []
                for m in msgs[-per_session:]:
                    body = content_text(m.get("content"))
                    if not body:
                        continue
                    ts = parse_ts(m.get("ts"))
                    last_msg_ts = ts or last_msg_ts
                    events.append(ev(ts, "cline", m.get("role", "?"), clip(body, 200)))
            except (OSError, json.JSONDecodeError):
                pass
        sessions.append(
            {
                "id": d.name,
                "status": meta.get("status"),
                "model": meta.get("model"),
                "provider": meta.get("provider"),
                "cost": (meta.get("metadata") or {}).get("totalCost"),
                "last_seen": last_msg_ts or (msgs_path.stat().st_mtime if msgs_path.exists() else None),
                "mtime": d.stat().st_mtime,
            }
        )
    return events, sessions


def read_codex(limit: int = 10, max_age_s: float = 86400) -> list[dict]:
    """Recent Codex rollout items only; older sessions are history, not progress."""
    events: list[dict] = []
    if not CODEX_SESSIONS.exists():
        return events
    try:
        files = sorted(CODEX_SESSIONS.rglob("rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:2]
    except OSError:
        return events
    cutoff = time.time() - max_age_s
    for f in files:
        try:
            if f.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_ts(o.get("timestamp"))
            if not ts or ts < cutoff:
                continue
            payload = o.get("payload") if isinstance(o.get("payload"), dict) else {}
            kind = str(o.get("type") or "?")
            if payload.get("type"):
                kind += "/" + str(payload["type"])
            text = ""
            for key in ("text", "message", "content", "command"):
                v = payload.get(key)
                if isinstance(v, str) and v.strip():
                    text = v
                    break
            events.append(ev(ts, "codex", kind, clip(text or kind, 160)))
    return events


def read_project(limit: int = 10) -> list[dict]:
    events: list[dict] = []
    if not PROJECT.exists():
        return events
    files = []
    for root, dirs, names in os.walk(PROJECT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for n in names:
            p = Path(root) / n
            try:
                files.append((p.stat().st_mtime, p, p.stat().st_size))
            except OSError:
                continue
    files.sort(reverse=True, key=lambda t: t[0])
    for mtime, p, size in files[:limit]:
        try:
            rel = p.relative_to(PROJECT)
        except ValueError:
            rel = p
        events.append(ev(mtime, "project", "file", f"{rel} ({size}B)"))
    return events


# -------------------------------------------------------------------- status
def codex_last_log() -> float | None:
    if not CODEX_LOGS.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{CODEX_LOGS}?mode=ro", uri=True, timeout=3)
        row = con.execute("select max(ts) from logs").fetchone()
        con.close()
        return float(row[0]) if row and row[0] else None
    except (sqlite3.Error, OSError, ValueError):
        return None


def codex_last_rollout() -> float | None:
    if not CODEX_SESSIONS.exists():
        return None
    try:
        files = list(CODEX_SESSIONS.rglob("rollout-*.jsonl"))
        return max((f.stat().st_mtime for f in files), default=None)
    except OSError:
        return None


def feed_last() -> float | None:
    if not FEED.exists():
        return None
    try:
        lines = FEED.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        for line in reversed(lines):
            try:
                return parse_ts(json.loads(line).get("ts"))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return None


# --------------------------------------------------------------------- procs
PROC_CMD = (
    "Get-CimInstance Win32_Process | "
    "Where-Object { $_.Name -match 'codex|pwsh|python|node' } | "
    "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
)

HINTS = [(r"\\\\codex\\.exe|codex-code-mode", "codex"), (r"EvoResearch", "本项目"), (r"cline", "cline")]


def read_procs() -> tuple[list[str], dict]:
    """Return (human rows, running flags)."""
    running = {"codex": False, "project": False, "cline": False}
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", PROC_CMD],
            capture_output=True, text=True, timeout=25, encoding="utf-8", errors="replace",
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ["(进程视图不可用)"], running
    if not out:
        return [], running
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return ["(进程视图解析失败)"], running
    if isinstance(data, dict):
        data = [data]
    rows = []
    for p in data:
        cmd = (p.get("CommandLine") or "").strip()
        if not cmd or re.search(r"Get-CimInstance|Win32_Process", cmd):
            continue
        label = None
        for pat, name in HINTS:
            if re.search(pat, cmd, re.I):
                label = name
                break
        if label == "codex":
            running["codex"] = True
        elif label == "本项目":
            running["project"] = True
            if "agents-watch" in cmd:
                rows.append(f"[本项目] pid={p.get('ProcessId')} {clip(cmd, 140)}")
                continue
        elif label == "cline":
            running["cline"] = True
        if label and label != "本项目":
            rows.append(f"[{label}] pid={p.get('ProcessId')} {clip(cmd, 140)}")
        elif label == "本项目":
            rows.append(f"[本项目] pid={p.get('ProcessId')} {clip(cmd, 140)}")
    return rows, running


# -------------------------------------------------------------------- render
def build(limit: int, with_procs: bool) -> dict:
    feed = read_feed()
    cline_events, sessions = read_cline()
    codex_events = read_codex()
    project_events = read_project()

    timeline = [e for e in (feed + cline_events + codex_events + project_events) if e["ts"]]
    timeline.sort(key=lambda e: e["ts"])

    procs, running = ([], {"codex": False, "project": False, "cline": False})
    if with_procs:
        procs, running = read_procs()

    cline_last = max((s["last_seen"] for s in sessions if s.get("last_seen")), default=None)

    return {
        "now": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project": str(PROJECT),
        "sessions": sessions,
        "timeline": timeline[-limit:],
        "procs": procs,
        "status": {
            "cline": {"last": cline_last, "running": running["cline"], "sessions": len(sessions)},
            "codex": {"last_log": codex_last_log(), "last_rollout": codex_last_rollout(), "running": running["codex"]},
            "feed": {"last": feed_last(), "running": running["project"]},
        },
    }


def live(flag: bool, ts: float | None, within: float = 600) -> str:
    if ts and time.time() - ts < within:
        return "活跃"
    return "闲置" if flag else "-"


def render(data: dict) -> str:
    out = []
    out.append("=" * 98)
    out.append(f"三方进度 · {data['now']} · {data['project']}")
    out.append("=" * 98)

    st = data["status"]
    out.append("\n[谁在干什么]")
    c = st["cline"]
    sess = data["sessions"][0] if data["sessions"] else {}
    cost = sess.get("cost")
    cost_s = f"${cost:.4f}" if isinstance(cost, (int, float)) else "-"
    out.append(
        f"  Cline  {live(c['running'], c['last']):<4}  最后动作 {age(c['last']):<6}  "
        f"会话 {sess.get('id', '-')}  status={sess.get('status')}  "
        f"model={sess.get('provider')}/{sess.get('model')}  cost={cost_s}"
    )
    x = st["codex"]
    out.append(
        f"  Codex  {live(x['running'], x['last_log']):<4}  运行日志 {age(x['last_log']):<6}  "
        f"最近 rollout {age(x['last_rollout'])}"
    )
    f = st["feed"]
    out.append(f"  进度源 {live(f['running'], f['last']):<4}  最后一行 {age(f['last'])}  ({FEED.name})")

    if data["procs"]:
        out.append("\n[现在在跑]")
        for p in data["procs"]:
            out.append("  " + p)

    out.append("\n[时间线]  时间 · 来源 · 事件 · 内容")
    if not data["timeline"]:
        out.append("  (还没有事件)")
    for e in data["timeline"]:
        out.append(f"  {stamp(e['ts'])} · {e['agent']:<10} · {e['kind']:<16} · {clip(e['text'], 140)}")

    return "\n".join(out)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="One timeline for every agent on this project.")
    ap.add_argument("-n", "--limit", type=int, default=40)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-procs", action="store_true")
    args = ap.parse_args()

    try:
        while True:
            data = build(args.limit, not args.no_procs)
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                if args.follow:
                    os.system("cls" if os.name == "nt" else "clear")
                print(render(data))
            if not args.follow:
                return 0
            sys.stdout.flush()
            time.sleep(max(args.interval, 1.0))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
