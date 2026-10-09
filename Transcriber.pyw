"""Double-click launcher (no console window)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transcript_app.gui import main  # noqa: E402

sys.exit(main())
