"""Windows current-user DPAPI encryption. Never falls back to plaintext."""
import ctypes,json,os,time,uuid
from pathlib import Path
from ctypes import wintypes

class CredentialStoreError(RuntimeError): pass
class Blob(ctypes.Structure):
    _fields_=[('length',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]

def protect(raw,decrypt=False):
    if os.name!='nt': raise CredentialStoreError('当前系统不支持 Windows 账户加密，未保存密钥。')
    buffer=ctypes.create_string_buffer(raw)
    source=Blob(len(raw),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte)))
    output=Blob()
    crypt=ctypes.WinDLL('crypt32',use_last_error=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
    if decrypt:
        fn=crypt.CryptUnprotectData
        fn.argtypes=[ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
        args=(ctypes.byref(source),None,None,None,None,1,ctypes.byref(output))
    else:
        fn=crypt.CryptProtectData
        fn.argtypes=[ctypes.POINTER(Blob),wintypes.LPCWSTR,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
        args=(ctypes.byref(source),'EvoResearch model credentials',None,None,None,1,ctypes.byref(output))
    fn.restype=wintypes.BOOL
    if not fn(*args): raise CredentialStoreError('Windows 密钥加密或解密失败，请在同一 Windows 账户下重新配置。')
    try: return ctypes.string_at(output.data,output.length)
    finally: kernel.LocalFree(ctypes.cast(output.data,ctypes.c_void_p))

class CredentialStore:
    def __init__(self,path): self.path=Path(path)
    def load(self):
        if not self.path.exists():return {}
        try:
            raw=self.path.read_bytes()
            if len(raw)>16384:raise ValueError()
            data=json.loads(protect(raw,decrypt=True))
            if not isinstance(data,dict) or any(k not in ('DEEPSEEK_API_KEY','JEV_API_KEY') or not isinstance(v,str) or not 8<=len(v)<=512 or not v.isascii() or any(c.isspace() for c in v) for k,v in data.items()): raise ValueError()
            return data
        except Exception:
            raise CredentialStoreError('已保存的密钥无法读取；请使用原 Windows 账户或在设置中重新保存。') from None
    def save(self,data):
        encrypted=protect(json.dumps(data).encode('utf-8'))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            temp.write_bytes(encrypted)
            for attempt in range(5):
                try: temp.replace(self.path);return
                except PermissionError:
                    if attempt==4:raise
                    time.sleep(.025*(attempt+1))
        except OSError:
            raise CredentialStoreError('密钥加密保存失败，原配置未变更。') from None
        finally: temp.unlink(missing_ok=True)
