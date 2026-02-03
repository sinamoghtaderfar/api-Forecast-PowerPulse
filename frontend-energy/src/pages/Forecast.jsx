// src/pages/Forecast.jsx
import { useEffect, useState } from "react";
import { getAllForecasts, getCombinedForecast } from "../api/energyApi";

function Forecast() {
  
  const [data, setData] = useState({
    prophet: [],
    random_forest: [],
    xgboost: []
  });
  const [combined, setCombined] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const allForecasts = await getAllForecasts(10);
        const combinedForecast = await getCombinedForecast(10);

        setData(allForecasts);
        setCombined(combinedForecast.combined);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return <p>Loading forecasts...</p>;
  if (error) return <p>Error: {error}</p>;

  return (
    <div style={{ padding: "20px" }}>
      <h1>Energy Forecast</h1>

      <h2>Prophet Forecast</h2>
      <pre>{JSON.stringify(data.prophet, null, 2)}</pre>

      <h2>Random Forest Forecast</h2>
      <pre>{JSON.stringify(data.random_forest, null, 2)}</pre>

      <h2>XGBoost Forecast</h2>
      <pre>{JSON.stringify(data.xgboost, null, 2)}</pre>

      <h2>Combined Forecast</h2>
      <pre>{JSON.stringify(combined, null, 2)}</pre>
    </div>
  );
}

export default Forecast;
