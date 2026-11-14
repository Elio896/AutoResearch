"""任务路由的测试 + 一份小标注集（用来算出真实 F1）。

分类器不能"感觉准"，得能量出来。这份标注集是我手工标的：
每一条都对应一个明确的意图，不模糊。
"""
import unittest

from task_router import classify_by_rules, classify_with_model, route

# (任务文本, 正确类别)
LABELED = [
    # 低级任务：数据操作，千问能做
    ('把这份 CSV 的缺失值用中位数填掉', 'low_task'),
    ('统计每列有多少空值', 'low_task'),
    ('把数据导出成 csv', 'low_task'),
    ('删掉重复行', 'low_task'),
    ('把类别列做 one-hot 编码', 'low_task'),
    ('筛出 target=1 的行', 'low_task'),
    ('读取文件并打印前五行', 'low_task'),

    # 决策类：要在选项之间选
    ('该用逻辑回归还是随机森林', 'decision'),
    ('连续三轮没涨了，要不要继续', 'decision'),
    ('哪个模型在验证集上更好', 'decision'),
    ('这个方案值得继续投入吗', 'decision'),
    ('该停止还是再试一次', 'decision'),
    ('优先调深度还是调正则', 'decision'),

    # 思考类：要分析/设计
    ('分析为什么验证分数掉了 0.3', 'thinking'),
    ('设计一个新的特征工程方案', 'thinking'),
    ('解释召回率为什么这么低', 'thinking'),
    ('诊断这次过拟合的原因', 'thinking'),
    ('提出一个能改善少数类召回的方法', 'thinking'),
    ('推断哪个特征导致了泄漏', 'thinking'),
]


def f1_per_class(labels: list[str], preds: list[str]) -> dict:
    classes = sorted(set(labels))
    out = {}
    for cls in classes:
        tp = sum(1 for y, p in zip(labels, preds) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(labels, preds) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(labels, preds) if y == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[cls] = {'precision': round(precision, 3), 'recall': round(recall, 3),
                    'f1': round(f1, 3), 'support': sum(1 for y in labels if y == cls)}
    return out


class RuleClassifierF1Tests(unittest.TestCase):
    def test_macro_f1_above_threshold(self):
        labels = [y for _, y in LABELED]
        preds = [classify_by_rules(t).category for t, _ in LABELED]
        report = f1_per_class(labels, preds)
        macro = sum(v['f1'] for v in report.values()) / len(report)
        print('\n=== 规则分类器 F1 ===')
        for cls, v in report.items():
            print(f"  {cls:10s} P={v['precision']:.2f} R={v['recall']:.2f} "
                  f"F1={v['f1']:.2f} (n={v['support']})")
        print(f'  macro F1 = {macro:.3f}')
        # 及格线：故意不设太高，因为这是规则基线；下面还有 few-shot 兜底
        self.assertGreaterEqual(macro, 0.80,
                                f'规则分类器 macro F1 只有 {macro:.3f}，低于 0.80')

    def test_no_category_is_collapsed(self):
        """任何一类都不能被完全吞掉——否则等于没有这个类别。"""
        preds = [classify_by_rules(t).category for t, _ in LABELED]
        for cls in ('low_task', 'decision', 'thinking'):
            self.assertGreater(preds.count(cls), 0, f'{cls} 一条都没被分到')


class RouterTests(unittest.TestCase):
    def test_decision_goes_to_jev(self):
        result = route('该用逻辑回归还是随机森林')
        self.assertEqual(result['executor'], 'jev')

    def test_thinking_goes_to_deepseek(self):
        result = route('分析为什么验证分数掉了 0.3')
        self.assertEqual(result['executor'], 'deepseek')

    def test_low_task_stays_with_qwen(self):
        result = route('统计每列有多少空值')
        self.assertEqual(result['executor'], 'qwen')

    def test_low_task_downgrades_when_signal_says_no(self):
        """方案 A 验证不过时，低级任务降级——因为那说明千问其实做不好。"""
        from signal_router import SignalState
        bad_state = SignalState(stalled_rounds=3,
                                recent_change_kinds=['tune_capacity'],
                                historical_delta_by_kind={'tune_capacity': -0.02})
        result = route('把缺失值填掉', signal_state=bad_state)
        self.assertEqual(result['executor'], 'fallback')
        self.assertIn('方案 A 验证不过', result['reason'])

    def test_low_task_passes_when_signal_agrees(self):
        from signal_router import SignalState
        good_state = SignalState(stalled_rounds=0,
                                 recent_change_kinds=['tune_capacity'],
                                 historical_delta_by_kind={'tune_capacity': 0.05})
        result = route('把缺失值填掉', signal_state=good_state)
        self.assertEqual(result['executor'], 'qwen')

    def test_model_agreement_raises_confidence(self):
        def fake_proposer(messages, timeout=None):
            return {'message': {'content': 'decision'}}
        result = classify_with_model('该用哪个模型', proposer=fake_proposer)
        self.assertEqual(result.category, 'decision')
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_falls_back_to_rules_when_model_fails(self):
        def broken_proposer(messages, timeout=None):
            raise RuntimeError('model down')
        result = classify_with_model('统计空值个数', proposer=broken_proposer)
        self.assertEqual(result.category, 'low_task')   # 规则兜底
        self.assertIn(result.category, ('low_task', 'decision', 'thinking'))


if __name__ == '__main__':
    unittest.main()
