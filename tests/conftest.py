import sys
from pathlib import Path

# The hub is a flat folder of modules (camera_hub.py imports them by name), not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
