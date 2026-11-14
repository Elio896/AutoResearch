"""服务端生成折线图 SVG——不依赖前端 JS，保证一定能看见。

妈妈反复说图不出来。前端渲染链条上任何一环出错都会白屏，
所以这里再给一条路：后端直接把 SVG 画好返回，浏览器只需要显示它。
"""
from __future__ import annotations


def line_chart_svg(scores: list[float], best: float | None = None,
                   width: int = 720, height: int = 240,
                   title: str = '') -> str:
    """把一串分数画成 SVG。纯字符串拼接，不引入任何依赖。"""
    pad_l, pad_r, pad_t, pad_b = 52, 18, 34, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
             f'width="100%" height="{height}" style="background:#faf8f5">']

    if title:
        parts.append(f'<text x="{pad_l}" y="20" font-size="13" fill="#6b6560" '
                     f'font-family="sans-serif">{title}</text>')

    if not scores:
        parts.append(f'<text x="{width/2}" y="{height/2}" text-anchor="middle" '
                     f'font-size="14" fill="#9a928a" font-family="sans-serif">还没有数据</text>')
        parts.append('</svg>')
        return ''.join(parts)

    y_max = max(1.0, max(scores + ([best] if best is not None else [])))
    y_max = min(1.0, y_max)

    def x_of(i: int) -> float:
        if len(scores) == 1:
            return pad_l + plot_w / 2
        return pad_l + (i / (len(scores) - 1)) * plot_w

    def y_of(v: float) -> float:
        return pad_t + plot_h - (v / y_max) * plot_h

    # 横向网格 + 纵轴刻度
    for k in range(5):
        v = y_max * k / 4
        yy = y_of(v)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width - pad_r}" y2="{yy:.1f}" '
                     f'stroke="#eeeae5" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{yy + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="#9a928a" font-family="sans-serif">{v:.2f}</text>')

    # 历史最优阶梯
    if best is not None:
        running = -1.0
        pts = []
        for i, s in enumerate(scores):
            running = max(running, s)
            pts.append(f'{x_of(i):.1f},{y_of(running):.1f}')
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#15803d" '
                     f'stroke-width="1.5" stroke-dasharray="5 4" opacity="0.7"/>')

    # 实际折线
    pts = ' '.join(f'{x_of(i):.1f},{y_of(s):.1f}' for i, s in enumerate(scores))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#b45309" '
                 f'stroke-width="2.5" stroke-linejoin="round"/>')

    # 点 + 数值 + 横轴标签
    for i, s in enumerate(scores):
        is_best = best is not None and abs(s - best) < 1e-9
        color = '#15803d' if is_best else '#b45309'
        parts.append(f'<circle cx="{x_of(i):.1f}" cy="{y_of(s):.1f}" r="{5 if is_best else 4}" '
                     f'fill="{color}" stroke="#fff" stroke-width="1.5"/>')
        parts.append(f'<text x="{x_of(i):.1f}" y="{y_of(s) - 10:.1f}" text-anchor="middle" '
                     f'font-size="11" font-weight="600" fill="{color}" '
                     f'font-family="sans-serif">{s:.3f}</text>')
        parts.append(f'<text x="{x_of(i):.1f}" y="{height - 10}" text-anchor="middle" '
                     f'font-size="11" fill="#9a928a" font-family="sans-serif">v{i + 1}</text>')

    parts.append('</svg>')
    return ''.join(parts)
