import { getAllForecasts } from "../api/energyApi";
import { useEffect, useState } from "react";

function Forecast() {
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      const result = await getAllForecasts(10); // عدد 10 برای 10 سال آینده
      setData(result);
    };
    fetchData();
  }, []);

  return (
    <div>
      <h1>Energy Forecast</h1>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

export default Forecast;
