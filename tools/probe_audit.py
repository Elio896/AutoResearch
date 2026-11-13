import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evoundo import VersionChain   # noqa: E402

chain = VersionChain(run_id='probe')

# 1) 未知 parent
try:
    chain.add({'model': 'lr'}, 0.5, parent_id='does_not_exist')
    print('unknown parent: ACCEPTED (bug)')
except ValueError as exc:
    print('unknown parent: rejected ->', exc)

# 2) 嵌套引用共享
chain2 = VersionChain(run_id='probe2')
proposal = {'model': 'rf', 'parameters': {'depth': 3}}
v = chain2.add(proposal, 0.5)
proposal['parameters']['depth'] = 99
print('alias check: stored =', v.proposal['parameters']['depth'], '(3 = ok, 99 = bug)')

# 3) NaN / inf
for bad in (float('nan'), float('inf')):
    try:
        chain3 = VersionChain(run_id='probe3')
        chain3.add({'model': 'lr'}, bad)
        print(f'score {bad}: ACCEPTED (bug)')
    except ValueError:
        print(f'score {bad}: rejected ok')
