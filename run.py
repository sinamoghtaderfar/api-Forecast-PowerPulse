"""
Entry point for the PowerPulse FastAPI server.

Configuration (environment variables): POWERPULSE_HOST (default 127.0.0.1),
POWERPULSE_PORT (default 8000), POWERPULSE_RELOAD (default off; enable only
for development) and POWERPULSE_LOG_LEVEL (default info).
"""

import logging

import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    uvicorn.run(
        app="app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
    )
