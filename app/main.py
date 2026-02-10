from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.prophet_service import ProphetForecastService
from app.services.random_forest_service import RandomForestForecastService
from app.services.xgboost_service import XGBoostForecastService

from app.controllers.energy_controller import router as energy_router

app = FastAPI(title="Germany Energy Forecast API")

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


prophet_service = ProphetForecastService()
random_forest_service = RandomForestForecastService()
xgboost_service = XGBoostForecastService()

@app.on_event("startup")
async def startup_event():
    print("Starting up Energy Forecast API...")
    
    try:
        # Prophet
        print("Loading Prophet data...")
        prophet_service.load_data()
        prophet_service.train_model()
        print("Prophet model ready")

        # Random Forest
        print("Loading Random Forest data...")
        random_forest_service.load_data()
        random_forest_service.train_model()
        print("Random Forest model ready")

        # XGBoost
        print("⚡ Loading XGBoost data...")
        xgboost_service.load_data()
        xgboost_service.train_model()
        print(" XGBoost model ready")

        print(" All models loaded and ready for predictions!")

    except Exception as e:
        print(f" Error during startup: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Germany Energy Forecast API",
        "status": "running",
        "models": ["prophet", "random_forest", "xgboost"],
        "endpoints": {
            "prophet": "/forecast/prophet",
            "random_forest": "/forecast/random-forest", 
            "xgboost": "/forecast/xgboost",
            "all_models": "/forecast/all"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "api_version": "1.0.0"
    }

app.include_router(energy_router, prefix="/forecast")