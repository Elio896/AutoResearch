"""Local classification experiments. No LLM or network calls."""
import argparse
import hashlib
import json
import time
from pathlib import Path


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

    def _metrics(self, actual, predictions, prefix):
        from sklearn.metrics import f1_score, precision_score, recall_score
        macro = float(f1_score(actual, predictions, average='macro', zero_division=0))
        result = {'selection_metric': self.metric, 'pos_label': self.pos_label,
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

    def run(self, model='logistic_regression', parameters=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        if self.closed:
            raise RuntimeError('Session finalized; further tuning is forbidden.')
        self.ensure_ready()
        parameters = dict(parameters or {})
        allowed = {
            'logistic_regression': {'C', 'class_weight', 'max_iter'},
            'random_forest': {'n_estimators', 'max_depth', 'min_samples_leaf', 'class_weight'},
        }
        if model not in allowed or set(parameters) - allowed[model]:
            raise ValueError('Unsupported model or parameters.')
        if model == 'logistic_regression':
            options = {'max_iter': 1000, **parameters}
            estimator = LogisticRegression(random_state=self.seed, **options)
        else:
            options = {'n_estimators': 100, **parameters}
            if not 1 <= options['n_estimators'] <= 500:
                raise ValueError('n_estimators must be between 1 and 500.')
            estimator = RandomForestClassifier(random_state=self.seed, n_jobs=1, **options)
        numeric = list(self.x.select_dtypes(include='number').columns)
        categorical = [column for column in self.x.columns if column not in numeric]
        preprocess = ColumnTransformer([
            ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                                  ('scale', StandardScaler())]), numeric),
            ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                                      ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical),
        ])
        pipeline = Pipeline([('preprocess', preprocess), ('model', estimator)])
        started = time.perf_counter()
        pipeline.fit(self.x.iloc[self.train_indices], self.y.iloc[self.train_indices])
        predictions = pipeline.predict(self.x.iloc[self.validation_indices])
        actual = self.y.iloc[self.validation_indices]
        experiment_id = f'exp_{len(self.experiments) + 1:03d}'
        result = {'status': 'success', 'experiment_id': experiment_id,
                  'model': model, 'parameters': options, 'seed': self.seed,
                  'dataset_sha256': self.dataset_sha256, 'split_sizes': self.split_sizes,
                  'excluded_columns': self.risk_report['excluded_columns'],
                  'feature_review': self.risk_report['decisions'],
                  'validation_f1': float(f1_score(actual, predictions, average='macro', zero_division=0)),
                  'validation_accuracy': float(accuracy_score(actual, predictions)),
                  'duration_seconds': time.perf_counter() - started}
        result.update(self._metrics(actual, predictions, 'validation'))
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
