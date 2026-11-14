"""One-shot credential comparison proof. Never records Authorization or key."""
import getpass,hashlib,io,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_router import JevClient,JevError,request
key=''.join(getpass.getpass('Jev key (hidden): ').split())
proof={'endpoint':'https://api.typesafe.ai/v1/systemone','method':'POST','key_identifier':key[:11]+'...'+key[-5:],'key_length':len(key),'key_sha256':hashlib.sha256(key.encode()).hexdigest()}
original=request.build_opener
captured={}
class Tap:
 def __init__(self,opener):self.opener=opener
 def open(self,req,timeout):
  response=self.opener.open(req,timeout=timeout)
  with response: raw=response.read(32769)
  if len(raw)>32768:raise JevError('Oversize response')
  if key.encode() in raw:raise JevError('Unexpected credential reflection; body withheld')
  captured['http_status']=response.status
  captured['request_body']=json.loads(req.data)
  captured['response_body']=json.loads(raw)
  return io.BytesIO(raw)
request.build_opener=lambda *handlers:Tap(original(*handlers))
try:
 result=JevClient(key)({'trigger':'connection_verification','validation_history':[],'remaining_experiments':1,'remaining_cloud_calls':1},timeout=20)
 proof.update(captured);proof['status']='connected'
 Path('artifacts/jev_connection_proof.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
 print(json.dumps(proof,indent=2))
except JevError as error:print(json.dumps({**proof,'status':'failed','reason':str(error)}))
