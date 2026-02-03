const API_BASE = "http://127.0.0.1:8000";

export async function getAllForecasts(yearsAhead = 10) {
  const res = await fetch(
    `${API_BASE}/forecast/all?years_ahead=${yearsAhead}`
  );

  if (!res.ok) {
    throw new Error("Failed to fetch forecasts");
  }

  return await res.json();
}

export async function getCombinedForecast(yearsAhead = 10) {
  const res = await fetch(
    `${API_BASE}/forecast/combined?years_ahead=${yearsAhead}`
  );

  if (!res.ok) {
    throw new Error("Failed to fetch combined forecast");
  }

  return await res.json();
}
