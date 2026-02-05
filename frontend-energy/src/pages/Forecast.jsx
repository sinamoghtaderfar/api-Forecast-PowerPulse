import { useEffect, useState } from "react";
import { getAllForecasts } from "../api/energyApi";
import { Box, Typography } from "@mui/material";
import ProphetChart from "../components/Prophet Forecast/ProphetChart";
import ProphetTable from "../components/Prophet Forecast/ProphetTable";

function Forecast() {
  const [data, setData] = useState({ prophet: [], random_forest: [], xgboost: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const allForecasts = await getAllForecasts(10);
        setData(allForecasts);
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

  const prophetChartData = data.prophet?.map(item => ({
    year: item?.year ?? 0,
    forecast: item?.forecast ?? null,
    lower: item?.lower ?? null,
    upper: item?.upper ?? null
  })) ?? [];

  
const prophetTableData = data.prophet?.map(item => ({
  date: String(item.year),
  value: Number(item.forecast?.toFixed(1)) || null,
  lower: Number(item.lower?.toFixed(1))   || null,
  upper: Number(item.upper?.toFixed(1))   || null
})) ?? [];

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4" sx={{ mb: 2 }}>Prophet Forecast</Typography>

      
      <ProphetChart data={prophetChartData} />  

      {/* جدول Prophet */}
      <Typography variant="h5" sx={{ mt: 4, mb: 2 }}>Prophet Forecast Table</Typography>
      <ProphetTable data={prophetTableData} />  
    </Box>
  );
}

export default Forecast;
