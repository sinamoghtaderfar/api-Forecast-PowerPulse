// src/pages/ForecastXGBoost.jsx
import { useEffect, useState } from "react";
import { Box, CircularProgress, Alert } from "@mui/material";

import XGBoostChart from "../components/XGBoost Forecast/XGBoostChart";
import XGBoostTable from "../components/XGBoost Forecast/XGBoostTable";
import { getAllForecasts } from "../api/energyApi";

export default function ForecastXGBoost() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const allForecasts = await getAllForecasts(10);
        setData(allForecasts.xgboost);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading)
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
        <CircularProgress />
      </Box>
    );

  if (error)
    return (
      <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "50vh" }}>
        <Alert severity="error">Error: {error}</Alert>
      </Box>
    );
  return (
    <div style={{ padding: "20px" }}>
      <h1>XGBoost Forecast</h1>
      <XGBoostChart data={data} />
      <XGBoostTable data={data} />
    </div>
  );
}
