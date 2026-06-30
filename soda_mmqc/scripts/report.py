#!/usr/bin/env python3
"""Launch the evaluation reporting Streamlit app."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    os.environ.setdefault("STREAMLIT_SERVER_RUN_ON_SAVE", "false")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

    import streamlit.web.cli as stcli

    app_script = (
        Path(__file__).resolve().parent.parent / "reporting" / "streamlit_app.py"
    )
    sys.argv = [
        "streamlit",
        "run",
        str(app_script),
        "--server.headless=true",
        "--server.address=localhost",
        "--server.port=8502",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--global.developmentMode=false",
        "--server.fileWatcherType=none",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
