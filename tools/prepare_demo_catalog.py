"""Prepare public original catalog offline; preserve existing user datasets."""
import sys,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from local_catalog import LocalCatalog
catalog=LocalCatalog();fixture=root/'fixtures/catalog/customer_churn.csv'
if fixture.exists() and not any(r['id']=='customer_churn' for r in catalog.records()):
 row=json.loads(fixture.with_suffix('.json').read_text(encoding='utf-8'));raw=fixture.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Public fixture checksum mismatch')
 catalog.add(row['id'],raw.decode('utf-8'),row['target'],row['aliases'],row['provenance'])
print('Offline original public catalog prepared:',len(catalog.records()),'datasets')
