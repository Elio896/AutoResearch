"""Start the local demonstration and open it, without exposing a terminal."""
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

root = Path(__file__).resolve().parents[1]
port = int(os.environ.get('EVORESEARCH_WEB_PORT','8012'))
url = f'http://127.0.0.1:{port}/'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def ready():
    try:
        with opener.open(url+'ml/version', timeout=1) as response:
            return b'ml-chain-19' in response.read(1000)
    except Exception:
        return False


def message(text):
    if os.name == 'nt':
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, 'EvoResearch', 0)
    else:
        print(text)


def main():
    if ready():
        webbrowser.open(url)
        return
    if os.name == 'nt':
        restart = subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass',
                                  '-File',str(root/'tools/release_old_ml.ps1'),'-Port',str(port)],
                                 capture_output=True,timeout=15,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        if restart.returncode:
            message('端口 8012 被其他程序占用，未关闭该程序。请关闭旧研究服务后重试。')
            return
    candidates = [os.getenv('EVORESEARCH_PYTHON'), str(root/'.venv/Scripts/python.exe'),
                  r'C:\workspace\EvoResearch\.venv\Scripts\python.exe', shutil.which('python')]
    python = None
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        check = subprocess.run([candidate, '-c', 'import uvicorn,fastapi,pandas,sklearn'],
                               capture_output=True, timeout=20,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if check.returncode == 0:
            python = candidate
            break
    if python is None:
        message('未找到已安装项目依赖的 Python 环境。请提供现有环境路径，不会自动安装软件。')
        return
    (root/'artifacts').mkdir(exist_ok=True)
    log_path = root/'artifacts/showcase-startup.log'
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen([python,'-m','uvicorn','ml_api:create_app','--factory',
                                    '--app-dir',str(root),'--host','127.0.0.1','--port',str(port)],cwd=root,
                                   stdout=log,stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    for _ in range(40):
        if ready():
            webbrowser.open(url)
            return
        if process.poll() is not None:
            break
        time.sleep(.5)
    message('演示服务未能启动。错误已保存在：\n'+str(log_path)+'\n请将这个日志发给我，不需要重新配置 API。')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        message('启动遇到问题：'+type(exc).__name__+'。请将提示发给我。')
