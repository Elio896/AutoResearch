"""Rollback only the isolated Qwen demo, including its visible report."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qwen_demo import DemoStore
p=argparse.ArgumentParser();p.add_argument('--latest',action='store_true');p.add_argument('--clear',action='store_true');p.add_argument('--list',action='store_true');a=p.parse_args()
store=DemoStore(Path(__file__).resolve().parents[1]/'artifacts/ml/demo')
if a.clear:store.snapshot();store.reset();print('Only demo memory cleared; production untouched.')
elif a.latest:print(store.rollback()['restored'])
else:print('Demo snapshots:',len(list((store.root/'snapshots').glob('*'))))
