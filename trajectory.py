"""Append-only round records and a SHA-256 hash chain for research runs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


_ROUND_ID = re.compile(r"^r[0-9]{3}_[a-z0-9_]+$")
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def diff_proposals(prev: dict | None, curr: dict) -> list[dict]:
    """Return a deterministic field-level diff between model proposals."""
    if not isinstance(curr, dict):
        raise TypeError("curr must be a proposal dictionary.")
    if prev is None:
        return []
    if not isinstance(prev, dict):
        raise TypeError("prev must be a proposal dictionary or None.")

    changes = []
    previous_model, current_model = prev.get("model"), curr.get("model")
    if previous_model != current_model:
        changes.append({
            "op": "replace",
            "path": "model",
            "from": previous_model,
            "to": current_model,
        })

    previous_parameters = prev.get("parameters") or {}
    current_parameters = curr.get("parameters") or {}
    if not isinstance(previous_parameters, dict) or not isinstance(current_parameters, dict):
        raise TypeError("proposal parameters must be dictionaries.")
    for key in sorted(set(previous_parameters) | set(current_parameters)):
        path = f"parameters.{key}"
        if key not in previous_parameters:
            changes.append({"op": "add", "path": path, "to": current_parameters[key]})
        elif key not in current_parameters:
            changes.append({"op": "remove", "path": path, "from": previous_parameters[key]})
        elif previous_parameters[key] != current_parameters[key]:
            changes.append({
                "op": "change",
                "path": path,
                "from": previous_parameters[key],
                "to": current_parameters[key],
            })
    return changes


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def record_hash(record: dict) -> str:
    body = json.loads(json.dumps(record, ensure_ascii=False))
    body.get("chain", {}).pop("hash", None)
    return "sha256:" + hashlib.sha256(canonical_json(body)).hexdigest()


class TrajectoryWriter:
    """Write immutable ``round_record.json`` files and their ordered trace."""

    def __init__(self, run_dir: str | Path, run_id: str):
        if not isinstance(run_id, str) or len(run_id) < 3:
            raise ValueError("run_id must contain at least three characters.")
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.rounds_dir = self.run_dir / "rounds"
        self.trace_path = self.run_dir / "trace.jsonl"
        self._entries = self._read_existing()

    @property
    def last_round_id(self) -> str | None:
        return self._entries[-1]["round_id"] if self._entries else None

    @property
    def last_hash(self) -> str | None:
        return self._entries[-1]["hash"] if self._entries else None

    @property
    def next_round(self) -> int:
        return len(self._entries)

    def append(self, record: dict) -> dict:
        """Hash and persist a complete schema-shaped round record."""
        if not isinstance(record, dict):
            raise TypeError("round record must be a dictionary.")
        round_number = record.get("round")
        round_id = record.get("round_id")
        if type(round_number) is not int or round_number != self.next_round:
            raise ValueError(f"round must be the next sequence number ({self.next_round}).")
        if not isinstance(round_id, str) or not _ROUND_ID.fullmatch(round_id):
            raise ValueError("round_id does not match the trace specification.")
        if record.get("run_id") != self.run_id:
            raise ValueError("round record run_id does not match this writer.")
        if round_id in {entry["round_id"] for entry in self._entries}:
            raise ValueError(f"round_id already exists: {round_id}")

        payload = json.loads(json.dumps(record, ensure_ascii=False))
        payload["chain"] = {
            "algo": "sha256",
            "prev_hash": self.last_hash,
            "hash": "",
        }
        payload["chain"]["hash"] = record_hash(payload)

        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        round_dir = self.rounds_dir / round_id
        round_dir.mkdir(parents=True, exist_ok=False)
        record_path = round_dir / "round_record.json"
        temporary_path = round_dir / ".round_record.json.tmp"
        try:
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(record_path)
            trace_entry = {
                "round": round_number,
                "round_id": round_id,
                "prev_hash": payload["chain"]["prev_hash"],
                "hash": payload["chain"]["hash"],
            }
            with self.trace_path.open("a", encoding="utf-8", newline="\n") as trace:
                trace.write(json.dumps(trace_entry, ensure_ascii=False,
                                       separators=(",", ":")) + "\n")
                trace.flush()
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise

        self._entries.append(trace_entry)
        return payload

    def _read_existing(self) -> list[dict]:
        if not self.trace_path.exists():
            return []
        entries = []
        for line_number, line in enumerate(
                self.trace_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid trace JSON on line {line_number}.") from exc
            expected_round = len(entries)
            if (not isinstance(entry, dict)
                    or entry.get("round") != expected_round
                    or entry.get("prev_hash") != (entries[-1]["hash"] if entries else None)
                    or not isinstance(entry.get("round_id"), str)
                    or not _ROUND_ID.fullmatch(entry["round_id"])
                    or not isinstance(entry.get("hash"), str)
                    or not _HASH.fullmatch(entry["hash"])):
                raise ValueError(f"Broken trace chain on line {line_number}.")
            record_path = self.rounds_dir / entry["round_id"] / "round_record.json"
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Missing or invalid round record for line {line_number}.") from exc
            if (record.get("round") != expected_round
                    or record.get("round_id") != entry["round_id"]
                    or record.get("chain", {}).get("prev_hash") != entry["prev_hash"]
                    or record.get("chain", {}).get("hash") != entry["hash"]
                    or record_hash(record) != entry["hash"]):
                raise ValueError(f"Round record hash mismatch on line {line_number}.")
            entries.append(entry)
        return entries
