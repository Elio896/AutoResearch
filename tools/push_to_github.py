#!/usr/bin/env python3
"""push_to_github.py - create a private GitHub repo and upload the project, without git.

Why this exists: this machine has no git and no gh. The GitHub REST API is enough:
resolve the user, create the repo, upload one blob per file, build one tree, one
commit, then point refs/heads/main at it.

The token is never written to disk or printed. It is read from, in order:
  --token-file <path>   a file outside the repo
  $env:GITHUB_TOKEN     process environment
  GITHUB_TOKEN          user-level environment (registry)

Usage
    .venv\\Scripts\\python.exe tools\\push_to_github.py --dry-run
    .venv\\Scripts\\python.exe tools\\push_to_github.py --token-file C:\\workspace\\.gh_token
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
API = "https://api.github.com"

# Directories that must never be uploaded (local state, secrets, vendored copies).
SKIP_DIRS = {
    ".venv", "venv", "__pycache__", ".git", ".pytest_cache",
    "artifacts", "data", "incoming", "bundle_extract", "node_modules",
}
SKIP_FILES = {"research_agent_bundle.zip"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log", ".sqlite3", ".whl"}


def collect_files() -> list[Path]:
    out: list[Path] = []
    for root, dirs, names in os.walk(PROJECT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for n in sorted(names):
            p = Path(root) / n
            if n in SKIP_FILES or p.suffix in SKIP_SUFFIXES:
                continue
            out.append(p)
    return out


def read_token(args) -> str:
    if args.token_file:
        raw = Path(args.token_file).read_text(encoding="utf-8-sig").strip()
        if raw:
            return raw
    for scope in ("Process", "User"):
        try:
            v = os.environ.get("GITHUB_TOKEN") if scope == "Process" else None
            if not v and os.name == "nt":
                import winreg  # noqa: PLC0415
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                    v = winreg.QueryValueEx(k, "GITHUB_TOKEN")[0]
            if v and v.strip():
                return v.strip()
        except (OSError, ImportError, IndexError):
            continue
    return ""


def api(token: str, method: str, path: str, payload=None, timeout=60):
    url = path if path.startswith("http") else API + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(body) if body.strip() else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"message": body[:400]}
        return e.code, parsed


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-name", default="EvoResearch")
    ap.add_argument("--description", default="预算感知的自主科研 Agent：CSV 分类研究闭环（本地小模型 + 云端按需介入）")
    ap.add_argument("--token-file", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--public", action="store_true", help="default is private")
    args = ap.parse_args()

    files = collect_files()
    total = sum(p.stat().st_size for p in files)
    print(f"待上传 {len(files)} 个文件, {total / 1024:.1f} KB, 可见性={'public' if args.public else 'private'}")
    for p in files:
        print(f"  {p.relative_to(PROJECT).as_posix():<48} {p.stat().st_size:>9}")
    if args.dry_run:
        return 0

    token = read_token(args)
    if not token:
        print("\n[错误] 没找到 token。用 --token-file 指一个仓库外的文件，或设 GITHUB_TOKEN。")
        return 2

    status, me = api(token, "GET", "/user")
    if status != 200:
        print(f"\n[错误] token 无效或权限不足 (HTTP {status}): {me.get('message')}")
        return 2
    owner = me.get("login")
    print(f"\n已认证: {owner}")

    status, repo = api(token, "POST", "/user/repos", {
        "name": args.repo_name,
        "description": args.description,
        "private": not args.public,
        "auto_init": False,
        "has_issues": True,
        "has_wiki": False,
    })
    if status == 201:
        print(f"仓库已创建: {repo['html_url']}")
    elif status == 422:
        status2, repo = api(token, f"GET", f"/repos/{owner}/{args.repo_name}")
        if status2 != 200:
            print(f"[错误] 仓库已存在但读不到: {repo.get('message')}")
            return 2
        print(f"仓库已存在，继续推送: {repo['html_url']}")
    else:
        print(f"[错误] 建仓失败 (HTTP {status}): {repo.get('message')}")
        return 2

    print("上传 blob ...")
    tree = []
    for p in files:
        rel = p.relative_to(PROJECT).as_posix()
        content = base64.b64encode(p.read_bytes()).decode("ascii")
        status, blob = api(token, "POST", f"/repos/{owner}/{args.repo_name}/git/blobs",
                           {"content": content, "encoding": "base64"})
        if status != 201:
            print(f"  [失败] {rel}: HTTP {status} {blob.get('message')}")
            return 2
        tree.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        print(f"  ok {rel}")

    status, new_tree = api(token, "POST", f"/repos/{owner}/{args.repo_name}/git/trees",
                           {"tree": tree})
    if status != 201:
        print(f"[错误] 建 tree 失败: {new_tree.get('message')}")
        return 2

    status, commit = api(token, "POST", f"/repos/{owner}/{args.repo_name}/git/commits",
                         {"message": "Initial import: EvoResearch research workbench",
                          "tree": new_tree["sha"]})
    if status != 201:
        print(f"[错误] 建 commit 失败: {commit.get('message')}")
        return 2

    status, ref = api(token, "POST", f"/repos/{owner}/{args.repo_name}/git/refs",
                      {"ref": "refs/heads/main", "sha": commit["sha"]})
    if status != 201:
        status, ref = api(token, "PATCH", f"/repos/{owner}/{args.repo_name}/git/refs/heads/main",
                          {"sha": commit["sha"], "force": True})
        if status not in (200, 201):
            print(f"[错误] 更新 main 失败: {ref.get('message')}")
            return 2

    print(f"\n完成: {repo['html_url']}")
    print(f"commit: {commit['sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
