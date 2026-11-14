"""Prepare supported real datasets using existing dependencies only."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from local_catalog import LocalCatalog


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--churn',action='store_true',help='Fetch public OpenML 40701 only if the catalog does not already have it')
    args=parser.parse_args();catalog=LocalCatalog()
    if args.churn and not any(r['id']=='customer_churn' for r in catalog.records()):
        from data_source import OpenMLCatalog
        csv,target,source=OpenMLCatalog().download('40701')
        if target!='class':raise ValueError('Upstream target changed; review dataset before import')
        catalog.add('customer_churn',csv,target,['customer churn','churn','客户流失','电信客户','telecom'],source)
    print(json.dumps({'datasets':[{k:r[k] for k in ('id','rows','columns','target','sha256')} for r in catalog.records()],
                      'source':'Real sklearn bundled datasets and optional fixed public OpenML dataset; no synthetic fallback'},ensure_ascii=True))


if __name__=='__main__':main()
