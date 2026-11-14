"""Local classification experiments. No LLM or network calls."""
import argparse
import hashlib
import json
import time
from pathlib import Path


INT_PARAMETERS = {
    'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_iter',
    'n_jobs', 'random_state', 'max_leaf_nodes', 'min_samples_leaf', 'max_features_int'}


def coerce_integer_parameters(model, parameters):
    """Small models write 100.0 where sklearn needs 100; fix that, keep a record.

    Only parameters known to be integers are touched, and only when the value has no
    fractional part. Everything else is left alone so genuine errors still surface.
    """
    fixed = {}
    for key, value in dict(parameters or {}).items():
        if key in INT_PARAMETERS and isinstance(value, float) and value.is_integer():
            fixed[key] = int(value)
        else:
            fixed[key] = value
    return fixed


class ResearchSession:
    """One dataset, fixed splits, validation-only iteration, one final test.

    State is in memory: keep this object alive throughout a research task.
    Restart persistence and a service API are deliberately deferred.
    """

    def __init__(self, csv_path, target, seed=42, feature_decisions=None, metric="f1_macro", pos_label="1"): 
        import pandas as pd
        from sklearn.model_selection import train_test_split

        if metric not in ('f1_macro', 'f1_minority'):
            raise ValueError('Unsupported selection metric.')
        self.metric, self.pos_label = metric, str(pos_label)
        self.csv_path = csv_path          # 多种子重跑要用同一份数据
        self.feature_decisions = feature_decisions
        self.target = str(target)          # 轨迹要记目标列名，供审计
        raw = Path(csv_path).read_bytes()
        self.dataset_sha256 = hashlib.sha256(raw).hexdigest()
        from io import BytesIO
        frame = pd.read_csv(BytesIO(raw))
        if not frame.columns.is_unique or target not in frame.columns:
            raise ValueError('CSV must contain a unique target column.')
        if len(frame.columns) < 2 or frame.empty:
            raise ValueError('CSV needs rows and at least one feature column.')
        if frame[target].isna().any():
            raise ValueError('Target contains missing labels; correct the CSV first.')
        self.y = frame[target].astype(str)
        if self.y.nunique() < 2:
            raise ValueError('Classification requires at least two classes.')
        self.x = frame.drop(columns=[target])
        if self.x.select_dtypes(include='number').isin([float('inf'), -float('inf')]).any().any():
            raise ValueError('Numeric features contain infinity.')
        self.seed = seed
        indices = list(range(len(frame)))
        try:
            remaining, self.test_indices = train_test_split(
                indices, test_size=0.2, random_state=seed, stratify=self.y)
            self.train_indices, self.validation_indices = train_test_split(
                remaining, test_size=0.25, random_state=seed,
                stratify=self.y.iloc[remaining])
        except ValueError as error:
            raise ValueError('Not enough samples per class for stratified 60/20/20 splits.') from error
        from integrity import apply_review, inspect_features
        findings = inspect_features(self.x.iloc[self.train_indices], self.y.iloc[self.train_indices],
                                    self.dataset_sha256, target)
        self.x, self.risk_report = apply_review(self.x, findings, feature_decisions or {})
        # Numeric-looking string labels trigger a balanced-class-weight failure in
        # the installed sklearn forest implementation. Fit a label vocabulary on
        # training labels only; use stable integer class IDs for every estimator.
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder().fit(self.y.iloc[self.train_indices])
        if metric == 'f1_minority':
            counts = self.y.iloc[self.train_indices].value_counts()
            if len(counts) != 2 or self.pos_label not in counts.index:
                raise ValueError('Minority F1 requires binary labels and an explicit positive label.')
            if counts[self.pos_label] > counts.min():
                raise ValueError('Positive label must be a minority class in training data.')
        self.positive_id = (int(self.label_encoder.transform([self.pos_label])[0])
                            if self.pos_label in self.label_encoder.classes_ else None)
        self.y = pd.Series(self.label_encoder.transform(self.y), index=self.y.index)
        self.split_sizes = {'train': len(self.train_indices),
                            'validation': len(self.validation_indices),
                            'test': len(self.test_indices)}
        self.experiments = {}
        self.closed = False

    def planning_context(self):
        """Bounded aggregate evidence from training rows, with anonymized class IDs."""
        training = self.x.iloc[self.train_indices]
        counts = self.y.iloc[self.train_indices].value_counts().sort_index()
        numeric = len(training.select_dtypes(include='number').columns)
        return {'scope': 'training_only', 'rows': len(training), 'features': len(training.columns),
                'numeric_features': numeric, 'categorical_features': len(training.columns) - numeric,
                'missing_fraction': float(training.isna().sum().sum() / max(1, training.size)),
                'constant_features': int((training.nunique(dropna=False) <= 1).sum()),
                'class_count': len(counts),
                'class_counts': [{'class_id': int(key), 'count': int(value)} for key, value in counts.iloc[:20].items()],
                'class_counts_truncated': len(counts) > 20,
                'selection_metric': self.metric,
                'positive_class_id': self.positive_id if self.metric == 'f1_minority' else None}

    def run_multi_seed(self, model='logistic_regression', parameters=None,
                       seeds=(42, 7, 2026, 1, 13)):
        """同一配置在多个种子上各跑一次，用来判断"提升是不是噪声"。

        为什么要成对：候选和最优在**同一个种子**上比，数据划分带来的波动
        被抵消，剩下的才归因于改动本身。

        每次都是一个完整的训练+验证（真跑），所以 fits = len(seeds)，
        但对外记账仍算 1 次实验——调用方需要知道墙钟是 S 倍。
        """
        scores = []
        details = []
        for seed in seeds:
            clone = ResearchSession(
                self.csv_path, self.target,
                metric=self.metric, pos_label=self.pos_label, seed=seed,
            )
            result = clone.run(model, parameters or {})
            score = result.get('validation_score', result.get('validation_f1'))
            scores.append(float(score))
            details.append({'seed': seed, 'score': float(score),
                            'experiment_id': result.get('experiment_id')})
        return {
            'model': model, 'parameters': parameters or {},
            'seeds': list(seeds), 'per_seed': scores,
            'seed_mean': sum(scores) / len(scores) if scores else 0.0,
            'seed_std': (sum((x - sum(scores) / len(scores)) ** 2 for x in scores)
                         / (len(scores) - 1)) ** 0.5 if len(scores) > 1 else 0.0,
            'details': details,
            'fits': len(seeds),
            'note': f'{len(seeds)} 个种子各跑一次；记账算 1 次实验，墙钟是 {len(seeds)} 倍',
        }

    def _metrics(self, actual, predictions, prefix):
        from sklearn.metrics import f1_score, precision_score, recall_score
        macro = float(f1_score(actual, predictions, average='macro', zero_division=0))
        from sklearn.metrics import precision_recall_fscore_support
        import numpy as np
        labels = np.unique(actual)[:20]
        precision, recall, f1, support = precision_recall_fscore_support(actual, predictions, labels=labels, zero_division=0)
        result = {prefix + '_class_metrics': [{'class_id': int(label), 'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(n)} for label, p, r, f, n in zip(labels, precision, recall, f1, support)], 'selection_metric': self.metric, 'pos_label': self.pos_label,
                  prefix + '_score': macro}
        if self.metric == 'f1_minority':
            minority = float(f1_score(actual, predictions, pos_label=self.positive_id, zero_division=0))
            result.update({prefix + '_score': minority, prefix + '_f1_minority': minority,
                prefix + '_minority_precision': float(precision_score(actual, predictions, pos_label=self.positive_id, zero_division=0)),
                prefix + '_minority_recall': float(recall_score(actual, predictions, pos_label=self.positive_id, zero_division=0))})
        return result

    def ensure_ready(self):
        if self.risk_report['blocked']:
            raise RuntimeError('Feature risk review required before training: ' +
                               ', '.join(self.risk_report['unresolved_columns']))


    def run(self, model='logistic_regression', parameters=None, preprocessing=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import MissingIndicator, SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        if self.closed:
            raise RuntimeError('Session finalized; further tuning is forbidden.')
        self.ensure_ready()
        parameters = coerce_integer_parameters(model, parameters)
        # 预处理开关：二德子 2026-11-14 指出只调模型天花板很低，
        # 表格任务的收益大半在数据这一侧（她们实测 readmission 加 isna 指示
        # 变量后正例 F1 从 0.165 到 0.443）。这里只放可信的四个开关。
        prep = {
            'impute': 'median',              # median / mean / most_frequent
            'encode': 'onehot',              # onehot / ordinal
            'imbalance_strategy': 'none',    # none / class_weight
            'add_missing_indicator': False,  # 新增 isna 列
            **(preprocessing or {}),
        }
        allowed_prep = {
            'impute': {'median', 'mean', 'most_frequent'},
            'encode': {'onehot', 'ordinal'},
            'imbalance_strategy': {'none', 'class_weight'},
            'add_missing_indicator': {True, False},
        }
        if set(prep) - set(allowed_prep):
            raise ValueError('Unsupported preprocessing key.')
        for key, allowed_values in allowed_prep.items():
            if prep[key] not in allowed_values:
                raise ValueError(f'Unsupported value for {key}: {prep[key]}')
        # imbalance_strategy=class_weight 必须在建 estimator 之前落到 parameters 上，
        # 否则估器已经按旧参数造好了，改了也白改。
        if prep['imbalance_strategy'] == 'class_weight' and 'class_weight' not in parameters:
            parameters = {**parameters, 'class_weight': 'balanced'}
        allowed = {
            'logistic_regression': {'C', 'class_weight', 'max_iter'},
            'random_forest': {'n_estimators', 'max_depth', 'min_samples_leaf', 'class_weight'},
            'hist_gradient_boosting': {'max_iter', 'max_depth', 'min_samples_leaf',
                                       'learning_rate', 'class_weight'},
        }
        if model not in allowed or set(parameters) - allowed[model]:
            raise ValueError('Unsupported model or parameters.')
        if model == 'logistic_regression':
            options = {'max_iter': 1000, **parameters}
            estimator = LogisticRegression(random_state=self.seed, **options)
        elif model == 'hist_gradient_boosting':
            options = {'max_iter': 100, **parameters}
            if not 1 <= options['max_iter'] <= 1000:
                raise ValueError('max_iter must be between 1 and 1000.')
            estimator = HistGradientBoostingClassifier(random_state=self.seed, **options)
        else:
            options = {'n_estimators': 100, **parameters}
            if not 1 <= options['n_estimators'] <= 500:
                raise ValueError('n_estimators must be between 1 and 500.')
            estimator = RandomForestClassifier(random_state=self.seed, n_jobs=1, **options)
        numeric = list(self.x.select_dtypes(include='number').columns)
        categorical = [column for column in self.x.columns if column not in numeric]
        # 缺失指示变量交给 SimpleImputer 自己带出来（add_indicator=True），
        # 比单独塞一个 MissingIndicator 稳——后者和 StandardScaler 串在一起时
        # 维度对不上，会静默把分数打到 0。
        numeric_steps = [('impute', SimpleImputer(
            strategy=prep['impute'], keep_empty_features=True,
            add_indicator=bool(prep['add_missing_indicator'])))]
        numeric_steps.append(('scale', StandardScaler()))
        if prep['encode'] == 'ordinal':
            # 序数编码：类别直接映射成整数，树模型够用，维度不膨胀
            from sklearn.preprocessing import OrdinalEncoder
            categorical_steps = [
                ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ]
        else:
            categorical_steps = [
                ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                ('encode', OneHotEncoder(handle_unknown='ignore')),
            ]
        preprocess = ColumnTransformer([
            ('numeric', Pipeline(numeric_steps), numeric),
            ('categorical', Pipeline(categorical_steps), categorical),
        ])
        # imbalance_strategy 已经在建估器之前落进 parameters，这里不用再改
        pipeline = Pipeline([('preprocess', preprocess), ('model', estimator)])
        started = time.perf_counter()
        pipeline.fit(self.x.iloc[self.train_indices], self.y.iloc[self.train_indices])
        predictions = pipeline.predict(self.x.iloc[self.validation_indices])
        actual = self.y.iloc[self.validation_indices]
        experiment_id = f'exp_{len(self.experiments) + 1:03d}'
        result = {'status': 'success', 'experiment_id': experiment_id,
                  'model': model, 'parameters': options, 'preprocessing': dict(prep), 'seed': self.seed,
                  'dataset_sha256': self.dataset_sha256, 'split_sizes': self.split_sizes,
                  'excluded_columns': self.risk_report['excluded_columns'],
                  'feature_review': self.risk_report['decisions'],
                  # 人的决策之外，必须把机器发现一起带出来。
                  # 二德子 2026-11-14 指出：原来只输出 decisions，
                  # 不传 --feature-decisions 时就只剩 {}，而 chargeback_filed
                  # 其实已被标成 candidate——用户看到 success + 0.9546 却毫不知情。
                  # candidate 不阻断之后，这个出口更不能缺。
                  # 二德子 2026-11-14 补充：已被 exclude 的列不该继续告警——
                  # 它已经不参与训练了，再提示只会让人以为没处理。
                  'risk_warnings': [
                      {'column': f['column'], 'level': f['level'], 'signals': f['signals']}
                      for f in self.risk_report['findings']
                      if f['column'] not in set(self.risk_report['excluded_columns'])
                  ],
                  'risk_findings': self.risk_report['findings'],
                  'risk_blocked': self.risk_report['blocked'],
                  'validation_f1': float(f1_score(actual, predictions, average='macro', zero_division=0)),
                  'validation_accuracy': float(accuracy_score(actual, predictions)),
                  'duration_seconds': time.perf_counter() - started}
        result.update(self._metrics(actual, predictions, 'validation'))
        # validation_score 必须跟着 selection_metric 走。
        # 二德子 2026-11-14 实测：不平衡数据用 macro-F1 会被多数类撑高——
        # readmission 报 0.508 而正例 F1 只有 0.177，等于用错指标选模型。
        if self.metric == 'f1_minority':
            from sklearn.metrics import f1_score as _f1
            result['validation_score'] = float(_f1(
                actual, predictions, average='binary',
                pos_label=self.positive_id, zero_division=0))
        elif self.metric == 'pr_auc' and self.metric in ('pr_auc',):
            from sklearn.metrics import average_precision_score
            probabilities = pipeline.predict_proba(self.x.iloc[self.validation_indices])
            classes = list(pipeline.classes_)
            pos_index = classes.index(self.positive_id) if self.positive_id in classes else -1
            result['validation_score'] = float(
                average_precision_score((actual == self.positive_id).astype(int),
                                        probabilities[:, pos_index]))
        else:
            result['validation_score'] = result['validation_f1']
        result['selection_metric'] = self.metric
        self.experiments[experiment_id] = (pipeline, result)
        return result

    def finalize(self, experiment_id):
        from sklearn.metrics import accuracy_score, f1_score
        if self.closed:
            raise RuntimeError('Test evaluation is allowed only once per session.')
        pipeline, _ = self.experiments[experiment_id]
        # Close before accessing test data, including if evaluation fails.
        self.closed = True
        predictions = pipeline.predict(self.x.iloc[self.test_indices])
        actual = self.y.iloc[self.test_indices]
        return {**self._metrics(actual, predictions, 'test'), 'experiment_id': experiment_id,
                'test_f1': float(f1_score(actual, predictions, average='macro', zero_division=0)),
                'test_accuracy': float(accuracy_score(actual, predictions))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--target', required=True)
    parser.add_argument('--metric', choices=['f1_macro', 'f1_minority'], default='f1_macro')
    parser.add_argument('--pos-label', default='1')
    parser.add_argument('--model', choices=['logistic_regression', 'random_forest'], default='logistic_regression')
    parser.add_argument('--parameters', default='{}', help='JSON model parameters')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--feature-decisions', default='{}', help='Explicit JSON feature decisions with action and reason')
    parser.add_argument('--finalize', action='store_true', help='Evaluate reserved test set and close task')
    args = parser.parse_args()
    session = ResearchSession(args.csv, args.target, seed=args.seed, feature_decisions=json.loads(args.feature_decisions), metric=args.metric, pos_label=args.pos_label)
    result = session.run(args.model, json.loads(args.parameters))
    if args.finalize:
        result['final_evaluation'] = session.finalize(result['experiment_id'])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
