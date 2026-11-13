"""打交付包：只收源码、测试、文档，排除虚拟环境和产物。

用法：
  .venv\\Scripts\\python.exe tools\\make_package.py D:\\EvoResearch-night-20261114.zip
"""
import sys
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {'.venv', 'artifacts', '__pycache__', '.git', 'node_modules',
                '.pytest_cache', 'incoming', 'compare'}
EXCLUDE_SUFFIX = {'.pyc', '.pyo', '.log'}
EXCLUDE_NAMES = {'.env', 'experience.jsonl'}

root = Path(__file__).resolve().parent.parent
target = Path(sys.argv[1]) if len(sys.argv) > 1 else root.parent / 'EvoResearch-package.zip'

included = []
skipped = 0
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = set(rel.parts)
        if parts & EXCLUDE_DIRS:
            skipped += 1
            continue
        if path.suffix.lower() in EXCLUDE_SUFFIX:
            skipped += 1
            continue
        if path.name in EXCLUDE_NAMES:
            skipped += 1
            continue
        try:
            if path.stat().st_size > 2_000_000:      # 大文件单独判断
                skipped += 1
                continue
        except OSError:
            skipped += 1
            continue
        zf.write(path, rel.as_posix())
        included.append(rel.as_posix())

print(f'included {len(included)} files, skipped {skipped}')
print(f'size {target.stat().st_size / 1024:.1f} KB')
print(f'path {target}')
