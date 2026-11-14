"""Safe failure categories; never persist raw provider exceptions."""
import ssl
from urllib.error import URLError, HTTPError
from model_budget import ModelBudgetExceeded
from light_model import LightModelTransportError

def failure_kind(exc):
    if isinstance(exc,ModelBudgetExceeded): return 'budget_exhausted'
    if isinstance(exc,HTTPError):
        return 'network_failure' if exc.code in (408,429,500,502,503,504) else 'model_protocol_error'
    if isinstance(exc,ssl.SSLCertVerificationError): return 'model_protocol_error'
    if isinstance(exc,(LightModelTransportError,ssl.SSLError,URLError,TimeoutError,ConnectionError)):
        return 'network_failure'
    return 'model_protocol_error'
