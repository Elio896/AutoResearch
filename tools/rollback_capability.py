"""回退能力表：把某次快照复制回去，或清空重来。

用法：
  python rollback_capability.py --list
  python rollback_capability.py --snapshot capability-20261115-031500.sqlite3
  python rollback_capability.py --latest
  python rollback_capability.py --clear
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / 'artifacts' / 'ml' / 'capability.sqlite3'
SNAPSHOTS = ROOT / 'artifacts' / 'evolution' / 'snapshots'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--snapshot')
    parser.add_argument('--latest', action='store_true')
    parser.add_argument('--clear', action='store_true')
    args = parser.parse_args()

    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    files = sorted(SNAPSHOTS.glob('capability-*.sqlite3'))
    if args.list or not (args.snapshot or args.latest or args.clear):
        print('能力表：', MEMORY, '存在' if MEMORY.exists() else '不存在')
        print('快照 %d 个：' % len(files))
        for f in files[-10:]:
            print('  ', f.name, '%.1f KB' % (f.stat().st_size / 1024))
        return

    if args.clear:
        if MEMORY.exists():
            backup = SNAPSHOTS / ('capability-before-clear-' + MEMORY.stat().st_mtime.__str__().split('.')[0] + '.sqlite3')
            shutil.copy2(MEMORY, backup)
            MEMORY.unlink()
            print('已清空；清空前备份 ->', backup.name)
        else:
            print('本来就没有能力表')
        return

    source = None
    if args.snapshot:
        candidate = SNAPSHOTS / args.snapshot
        if not candidate.exists():
            raise SystemExit('找不到快照：' + str(candidate))
        source = candidate
    elif args.latest:
        if not files:
            raise SystemExit('没有可用快照')
        source = files[-1]

    if MEMORY.exists():
        keep = SNAPSHOTS / ('capability-before-rollback-' + str(int(MEMORY.stat().st_mtime)) + '.sqlite3')
        shutil.copy2(MEMORY, keep)
        print('当前能力表已备份 ->', keep.name)
    shutil.copy2(source, MEMORY)
    print('已回退到', source.name)


if __name__ == '__main__':
    main()
