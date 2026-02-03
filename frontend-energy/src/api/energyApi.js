// src/api/energyApi.js
const API_BASE = "";

export async function getAllForecasts(yearsAhead = 10) {
  const res = await fetch(`${API_BASE}forecast/all?years_ahead=${yearsAhead}`);

  if (!res.ok) {
    throw new Error("Failed to fetch forecasts");
  }
  //   const data = await res.json(); // فقط یک بار body را بخوان
  // console.log(data);  
  return await res.json();
}

export async function getCombinedForecast(yearsAhead = 10) {
  const res = await fetch(`${API_BASE}/forecast/combined?years_ahead=${yearsAhead}`);
  if (!res.ok) {
    throw new Error("Failed to fetch combined forecast");
  }
  return await res.json();
}
