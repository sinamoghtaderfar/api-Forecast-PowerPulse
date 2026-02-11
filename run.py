"""
This is the entry point for the FastAPI server of the PowerPulse project.
It starts the server using Uvicorn, sets the host and port, and enables
auto-reload for development purposes. Uncomment the workers and timeout
options to adjust performance settings if needed.
"""

import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(
        app="app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        # workers=2,
        # timeout_keep_alive=65,
    )
