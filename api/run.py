# run.py
from fastapi import FastAPI
from app.controllers.energy_controller import router as energy_router

app = FastAPI(title="Germany Energy Forecast API")

# Include controller routes
app.include_router(energy_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
