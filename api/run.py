# run.py
"""
اجرای سرور FastAPI پروژه
استفاده: python run.py
"""

import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(
        app="app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,                    # در توسعه فعال باشه
        log_level="info",
        # workers=2,                    # برای production فعال کن (کامنت شده)
        # timeout_keep_alive=65,        # اگر نیاز به تنظیم timeout داری
    )