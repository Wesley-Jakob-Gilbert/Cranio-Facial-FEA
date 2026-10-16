import sys
from pathlib import Path

# Import the package from this checkout (worktrees share one editable install).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
