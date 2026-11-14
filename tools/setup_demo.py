import sys,subprocess,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[1]
if a.check:
 import fastapi,uvicorn,pandas,sklearn,httpx
 print('Dependencies OK; no install performed.')
else:
 if sys.version_info<(3,11):raise SystemExit('Python 3.11+ required')
 env=root/'.venv'
 if not env.exists():subprocess.run([sys.executable,'-m','venv',str(env)],check=True)
 python=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
 subprocess.run([str(python),'-m','pip','install','-r',str(root/'requirements-api.txt')],check=True)
 subprocess.run([str(python),str(root/'tools/prepare_demo_catalog.py')],check=True)
 print('Dependencies installed. Start Ollama and run: ollama pull qwen2.5:3b-instruct-q4_K_M')
