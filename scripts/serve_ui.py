#!/usr/bin/env python3
"""Run the btkit API + SPA server with uvicorn."""

import os
import sys

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import uvicorn

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050

if __name__ == "__main__":
    uvicorn.run(
        "btkit.analysis.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="warning",
    )
