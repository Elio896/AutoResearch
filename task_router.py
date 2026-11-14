"""任务路由：千问负责判别，JEV 和 DeepSeek 并列处理它做不了的部分。

架构（2026-11-14 与妈妈对齐）：

    数据/任务进来
         ↓
    ① 千问判别：这个我能做吗？
         ├─ 能做（low_task：清洗、格式、筛查这类基础活）
         │      ↓
         │  ② 方案 A 验证：它是**真的**能做吗？
         │      ├─ 是 → 千问自己做（零成本）
         │      └─ 否 → 当成"做不了"，进第 ③ 步
         │
         └─ 不能做 → ③ 千问再分类：该交给谁？
                ├─ 要**决策** → JEV
                └─ 要**思考** → DeepSeek

三个角色的关系：
    千问       —— 干基础活 + 两轮判别
    JEV        —— 决策类（与 DeepSeek **并列**，不是它的门）
    DeepSeek   —— 思考类

方案 A（signal_router）的用途是**验证千问是不是真能做好**，
不是"值不值得花钱"——这一点之前理解错了，现在改正。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------- 三类任务的判别特征 ----------------

# 决策类：要在几个选项里做选择、要不要继续、哪个更好
DECISION_PATTERNS = [
    r'该不该', r'要不要', r'是否值得', r'选哪个',
    # "哪个模型更好" 里"哪个"和"更"之间隔着词，不能要求连续
    r'哪个.{0,12}更', r'哪个好', r'继续还是', r'停止还是',
    r'优先', r'取舍', r'权衡', r'比较.*方案',
    # "该用 X 还是 Y" / "用 X 好还是 Y 好" ——选模型的典型问法
    r'该用', r'用\S{1,12}还是', r'\S{1,12}还是\S{1,12}[好强]', r'换成',
    r'worth it', r'trade-?off',
    r'which.*better', r'should.*(stop|continue|use|switch)', r'choose|select|prefer',
]

# 思考类：要分析原因、设计新东西、解释现象
THINKING_PATTERNS = [
    r'为什么', r'怎么会', r'原因', r'分析', r'解释', r'推断',
    r'设计', r'提出', r'改进', r'优化方案', r'诊断', r'定位',
    r'假设', r'预测', r'推理',
    r'why', r'explain', r'analy[sz]e', r'design|propose|diagnose|hypothes',
]

# 低级任务：数据操作、格式转换、固定流程
LOW_TASK_PATTERNS = [
    r'清洗', r'去重', r'填充', r'格式', r'转换', r'提取列', r'排序', r'筛选',
    r'统计.*行', r'计数', r'画图', r'导出', r'保存', r'读取', r'拼接',
    r'clean|dedup|fill|format|convert|sort|filter|count|export|read|csv',
]


@dataclass
class RouteDecision:
    category: str            # low_task / decision / thinking
    confidence: float        # 规则判定的把握（0~1）
    reason: str
    signals: dict

    def to_dict(self) -> dict:
        return {'category': self.category, 'confidence': round(self.confidence, 3),
                'reason': self.reason, 'signals': self.signals}


def _count(patterns: list[str], text: str) -> int:
    hits = 0
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            hits += 1
    return hits


def classify_by_rules(task: str) -> RouteDecision:
    """纯规则分类，零成本、可解释。作为第一层，也是 few-shot 的兜底。"""
    text = str(task or '').strip()
    if not text:
        return RouteDecision('low_task', 0.0, '空任务，按最低级处理', {})

    decision_hits = _count(DECISION_PATTERNS, text)
    thinking_hits = _count(THINKING_PATTERNS, text)
    low_hits = _count(LOW_TASK_PATTERNS, text)
    total = decision_hits + thinking_hits + low_hits

    signals = {'decision_hits': decision_hits, 'thinking_hits': thinking_hits,
               'low_hits': low_hits, 'length': len(text)}

    if total == 0:
        # 没有任何特征：按"低级任务"处理（保守——低成本先试）
        return RouteDecision('low_task', 0.3, '没有明显特征，先按低级任务试', signals)

    if decision_hits > thinking_hits and decision_hits > low_hits:
        conf = 0.5 + 0.5 * (decision_hits / total)
        return RouteDecision('decision', min(conf, 0.95),
                             f'出现 {decision_hits} 个决策类特征词，像是要选/要不要', signals)

    if thinking_hits > decision_hits and thinking_hits > low_hits:
        conf = 0.5 + 0.5 * (thinking_hits / total)
        return RouteDecision('thinking', min(conf, 0.95),
                             f'出现 {thinking_hits} 个思考类特征词，像是要分析/设计', signals)

    conf = 0.5 + 0.5 * (low_hits / total)
    return RouteDecision('low_task', min(conf, 0.95),
                         f'出现 {low_hits} 个低级任务特征词，像是数据操作', signals)


# ---------------- 千问做第二层判别（few-shot） ----------------

FEW_SHOT = [
    ('把这份 CSV 的缺失值用中位数填掉', 'low_task'),
    ('统计一下每列有多少个空值', 'low_task'),
    ('该用逻辑回归还是随机森林', 'decision'),
    ('已经连续三轮没涨了，要不要继续', 'decision'),
    ('分析一下为什么验证分数掉这么多', 'thinking'),
    ('设计一个新的特征工程方案', 'thinking'),
]

CLASSIFY_PROMPT = (
    '把任务分成三类之一，只回一个词：\n'
    'low_task  = 数据清洗/格式转换/统计/筛选这类基础操作\n'
    'decision  = 要在选项之间做选择，或判断要不要继续\n'
    'thinking  = 要分析原因、设计方案、推理预测\n\n'
    '例子：\n'
    + '\n'.join(f'  任务：{t}\n  类别：{c}' for t, c in FEW_SHOT)
    + '\n\n只输出 low_task / decision / thinking 其中一个，不要解释。'
)


def classify_with_model(task: str, proposer=None, timeout: float = 30) -> RouteDecision:
    """第二层：让千问做 few-shot 分类。

    有模型就用模型；模型不可用或答得不像，退回规则分类。
    这样现场演示时模型挂了也不会卡住。
    """
    rules = classify_by_rules(task)
    if proposer is None:
        return rules

    try:
        response = proposer([
            {'role': 'system', 'content': CLASSIFY_PROMPT},
            {'role': 'user', 'content': f'任务：{task}'},
        ], timeout=timeout)
    except Exception:                     # noqa: BLE001 - 模型挂了就用规则
        return rules

    content = response if isinstance(response, str) else \
        ((response.get('message') or {}).get('content') or '')
    text = str(content).strip().lower()
    for category in ('low_task', 'decision', 'thinking'):
        if category in text:
            # 模型和规则一致时给更高把握
            agree = (category == rules.category)
            return RouteDecision(
                category, 0.85 if agree else 0.7,
                ('模型与规则一致' if agree else '模型与规则不一致，按模型判'),
                {'rules_said': rules.category, 'model_said': category})
    return rules          # 模型答得不像，退回规则


# ---------------- 主路由 ----------------

def route(task: str, proposer=None, signal_state=None,
          timeout: float = 30) -> dict:
    """完整路由：分类 → （可解时）方案 A 验证 → 分给对应执行者。

    返回：
        {'category', 'executor', 'reason', 'step', 'signals'}

    executor 取值：
        'qwen'      —— 千问自己做
        'jev'       —— 决策类，交给 JEV
        'deepseek'  —— 思考类，交给 DeepSeek
        'fallback'  —— 低级任务但方案 A 验证不过，降级给云端
    """
    decision = classify_with_model(task, proposer=proposer, timeout=timeout)

    # ① 非低级任务：直接分给并列的两个大模型
    if decision.category == 'decision':
        return {'category': 'decision', 'executor': 'jev',
                'reason': f'{decision.reason} → 决策类交给 JEV',
                'step': 'classify', 'signals': decision.signals}

    if decision.category == 'thinking':
        return {'category': 'thinking', 'executor': 'deepseek',
                'reason': f'{decision.reason} → 思考类交给 DeepSeek',
                'step': 'classify', 'signals': decision.signals}

    # ② 低级任务：自己先干，但要用方案 A 验证「真能干好吗」
    if signal_state is not None:
        try:
            from signal_router import decide as signal_decide
            verification = signal_decide(signal_state)
            if verification.get('action') == 'escalate':
                # 方案 A 说"干不好"——降级，当成做不了处理
                return {'category': 'low_task', 'executor': 'fallback',
                        'reason': f'方案 A 验证不过：{verification.get("reason")} → 降级给云端',
                        'step': 'verify', 'signals': {**decision.signals,
                                                       'signal_verification': verification}}
            return {'category': 'low_task', 'executor': 'qwen',
                    'reason': f'低级任务且方案 A 验证通过：{verification.get("reason")}',
                    'step': 'verify', 'signals': {**decision.signals,
                                                   'signal_verification': verification}}
        except Exception as exc:          # noqa: BLE001
            return {'category': 'low_task', 'executor': 'qwen',
                    'reason': f'方案 A 不可用（{str(exc)[:60]}），按低级任务试',
                    'step': 'verify', 'signals': decision.signals}

    return {'category': 'low_task', 'executor': 'qwen',
            'reason': f'{decision.reason} → 千问自己做',
            'step': 'classify', 'signals': decision.signals}


if __name__ == '__main__':
    import json
    cases = [
        '把这份 CSV 的缺失值用中位数填掉',
        '统计每列有多少空值',
        '该用逻辑回归还是随机森林',
        '连续三轮没涨了，要不要继续',
        '分析为什么验证分数掉了 0.3',
        '设计一个新的特征工程方案',
    ]
    for case in cases:
        result = route(case)
        print(f'{case[:28]:30s} → {result["category"]:10s} {result["executor"]}')
