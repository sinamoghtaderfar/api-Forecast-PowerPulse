"""
Entry point for the PowerPulse FastAPI server.

Host, port and auto-reload can be overridden with the POWERPULSE_HOST,
POWERPULSE_PORT and POWERPULSE_RELOAD environment variables.
"""

import logging
import os

import uvicorn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        app="app.main:app",
        host=os.getenv("POWERPULSE_HOST", "0.0.0.0"),
        port=int(os.getenv("POWERPULSE_PORT", "8000")),
        reload=os.getenv("POWERPULSE_RELOAD", "1") == "1",
        log_level="info",
    )
