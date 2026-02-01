import pandas as pd
from prophet import Prophet



class ProphetForecastService:
    def __init__(self, data_path="app/data/germany-energy-clean.csv"):
        self.data_path = data_path
        self.df = None
        self.model = None

    def load_data(self):
        # load Data 
        self.df = pd.read_csv(self.data_path)
        self.df['ds'] = pd.to_datetime(self.df['year'], format='%Y')
        self.df = self.df.rename(columns={'electricity_generation': 'y'})
        print("Data loaded successfully")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        print("\nFirst 5 rows:")
        print(self.df.head())

    def train_and_forecast(self, years_ahead=10):
        if self.df is None:
            raise ValueError("Data not loaded! Call load_data() first.")
        # add variable Prophet
        self.model = Prophet()
        self.model.add_regressor('population')
        self.model.add_regressor('gdp')
        self.model.add_regressor('renewables_share_energy')

        self.model.fit(self.df)

        # future dataframe
        future = self.model.make_future_dataframe(periods=years_ahead, freq='YE')

        last_row = self.df.iloc[-1]

        future['population'] = last_row['population'] * (1 + 0.01) ** (future.index - self.df.index[-1])
        future['gdp'] = last_row['gdp'] * (1 + 0.02) ** (future.index - self.df.index[-1])
        future['renewables_share_energy'] = last_row['renewables_share_energy'] * (1 + 0.05) ** (future.index - self.df.index[-1])

        forecast = self.model.predict(future)

        print("\nProphet Forecast Result:")
        print(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(years_ahead))

        forecast_json = []
        for _, row in forecast.iterrows():
            forecast_json.append({
                "year": row['ds'].year,
                "forecast": row['yhat'],
                "lower": row['yhat_lower'],
                "upper": row['yhat_upper']
            })

        #print("\nForecast Result JSON:")
        #print(forecast_json)
        print("\nProphet Forecast JSON Output:")
        for item in forecast_json:
            print(f"{item}")
        return forecast_json
