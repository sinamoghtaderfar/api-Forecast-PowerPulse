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

prophet_service = ProphetForecastService("app/data/germany-energy-clean.csv")
random_forest_service = RandomForestForecastService("app/data/germany-energy-clean.csv")
xgboost_service = XGBoostForecastService("app/data/germany-energy-clean.csv")

@app.on_event("startup")
async def startup_event():
    print("=== شروع فرآیند startup ===")
    
    try:
        # Prophet
        prophet_service.load_data()
        prophet_service.train_model()          

        # Random Forest
        random_forest_service.load_data()
        random_forest_service.train_model()

        # XGBoost
        xgboost_service.load_data()
        xgboost_service.train_model()

    
    except Exception as e:
        print(f"{e}")
        

app.include_router(energy_router, prefix="/forecast")
