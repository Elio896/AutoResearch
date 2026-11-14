"""Reproducible two-round benchmark. Rules mode never makes a network request."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_system import ResearchCampaign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('rules', 'single_model'), default='rules')
    parser.add_argument('--max-calls', type=int, default=36)
    parser.add_argument('--max-tokens', type=int, default=150000)
    parser.add_argument('--max-seconds', type=int, default=180)
    parser.add_argument('--max-fits', type=int, default=80)
    parser.add_argument('--development-csv')
    parser.add_argument('--target', default='target')
    parser.add_argument('--corrections-json', help='Exported protocol.json or feedback snapshot to replay user constraints.')
    args = parser.parse_args()
    client = None
    if args.mode == 'single_model':
        from light_model import from_env
        client = from_env()
        if client is None:
            parser.error('Set LIGHT_MODEL_API_KEY or DEEPSEEK_API_KEY on the server.')
        client.max_calls = args.max_calls
    correction_store = None
    if args.corrections_json:
        from corrections import CorrectionStore
        payload = json.loads(Path(args.corrections_json).read_text(encoding='utf-8'))
        rows = payload if isinstance(payload, list) else payload.get('user_corrections', payload.get('rules', payload.get('records')))
        if not isinstance(rows, list):
            parser.error('Correction snapshot must contain a list of checked constraints.')
        correction_store = CorrectionStore(Path(args.output) / 'replayed-corrections.sqlite3')
        for row in rows:
            if row.get('status', 'active') != 'active':
                continue
            restored = correction_store.add(row['text'], kind=row['kind'], value=row.get('value'),
                                            scope=row['scope'], dataset_sha256=row.get('dataset_sha256'))
            if restored['status'] != 'active':
                parser.error('Correction snapshot contains an unsupported or conflicting rule.')
    report = ResearchCampaign(args.output, mode=args.mode, client=client,
                              max_calls=args.max_calls, max_tokens=args.max_tokens,
                              max_seconds=args.max_seconds, max_fits=args.max_fits,
                              correction_store=correction_store).run(args.development_csv, args.target)
    print(json.dumps({'rounds': [{'round': row['round'], 'decision': row['decision'],
                                 'score_delta': row.get('score_delta')} for row in report['rounds']],
                      'training_fits': report['training_fits'],
                      'elapsed_seconds': report['elapsed_seconds'],
                      'model_usage': {k: v for k, v in report['model_usage'].items() if k != 'records'},
                      'final_test': report['final_test']}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
