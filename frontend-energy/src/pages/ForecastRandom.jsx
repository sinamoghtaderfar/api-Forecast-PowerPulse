// src/pages/ForecastRandomForest.jsx
import { useEffect, useState } from "react";
import { Box, CircularProgress, Alert } from "@mui/material";

import RandomForestChart from "../components/Random Forecast/RandomForestChart";
import RandomForestTable from "../components/Random Forecast/RandomForestTable";
import { getAllForecasts } from "../api/energyApi";

export default function ForecastRandomForest() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const allForecasts = await getAllForecasts(10);
        setData(allForecasts.random_forest); 
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
      <h1>Random Forest Forecast</h1>
      <RandomForestChart data={data} />
      <RandomForestTable data={data} />
    </div>
  );
}
