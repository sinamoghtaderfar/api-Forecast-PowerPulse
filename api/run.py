
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