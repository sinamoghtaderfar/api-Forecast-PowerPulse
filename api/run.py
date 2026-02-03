# run.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.controllers.energy_controller import router as energy_router

app = FastAPI(title="Germany Energy Forecast API")

# CORS config (for React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(energy_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
