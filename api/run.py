from app.services.forecast_service import EnergyDataService

if __name__ == "__main__":
    service = EnergyDataService()
    df = service.load_data()
