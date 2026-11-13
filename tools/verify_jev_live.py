"""One explicit Jev connection check. Secret stays in process memory."""
import getpass,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_router import JevClient,JevError

def main():
    key=''.join(getpass.getpass('Jev key (hidden; memory only): ').split())
    client=JevClient(key)
    del key
    try:
        result=client({'trigger':'connection_verification','validation_history':[],
                       'remaining_experiments':1,'remaining_cloud_calls':1},timeout=20)
        print(json.dumps({'status':'connected',**result}),flush=True)
    except JevError as error:
        print(json.dumps({'status':'failed','reason':str(error)}),flush=True)
        return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
