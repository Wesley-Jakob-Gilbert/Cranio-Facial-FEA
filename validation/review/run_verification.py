"""Run the unmodified verification script with outputs confined to validation/."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("review_verify_fem", ROOT / "scripts/verify_fem.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.FIG_DIR = ROOT / "validation/review/figures"
module.FIG_DIR.mkdir(exist_ok=True)
sys.argv = [str(ROOT / "scripts/verify_fem.py"), "--json",
            str(ROOT / "validation/review/verification-full.json")]
module.main()
