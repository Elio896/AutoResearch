"""自检：demo 页面里每个被 JS 引用的元素 id 是否真的存在。

妈妈反复说"图不出来"，多半是某处 $() 拿到 null 抛异常，
把后面的渲染全带崩了。这个脚本静态检查一遍，不靠猜。
"""
import re
from pathlib import Path

HTML = Path('static/demo.html')
text = HTML.read_text(encoding='utf-8')

# 页面上真实存在的 id
ids_in_html = set(re.findall(r'\bid="([^"]+)"', text))

# JS 里被 $('xxx') 引用的 id
ids_used = set(re.findall(r"\$\('([^']+)'\)", text))

missing = sorted(ids_used - ids_in_html)
unused = sorted(ids_in_html - ids_used)

print(f'HTML 定义 {len(ids_in_html)} 个 id')
print(f'JS 引用 {len(ids_used)} 个 id')
print()
if missing:
    print('!!! JS 引用了但页面没有的 id（会导致 null 异常）:')
    for name in missing:
        print('   ', name)
else:
    print('OK: JS 引用的 id 全部存在')

if unused:
    print()
    print('（页面里定义了但 JS 没用到的 id，供参考）')
    for name in unused[:10]:
        print('   ', name)

raise SystemExit(1 if missing else 0)
