"""Original Qwen demo entry; model and data preserved."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qwen_demo import cli
if __name__=='__main__':cli('clean')
