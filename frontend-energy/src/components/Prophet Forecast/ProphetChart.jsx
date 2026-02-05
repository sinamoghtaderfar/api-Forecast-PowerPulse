// src/components/ProphetChart.jsx
import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";

export default function ProphetChart({ data }) {
  // data = [{ year: 2026, forecast: 504, lower: 483, upper: 523 }, ...]
  // map 
const chartData = data.map(item => ({
    date: item.year,
    value: item.forecast,
    lower: item.lower,
    upper: item.upper
  }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        {/* Lower */}
        <Line
          type="monotone"
          dataKey="lower"
          name="Lower Bound"
          stroke="#ef4444" 
          strokeWidth={1.5}
          dot={false}
        />
        {/* Forecast */}
        <Line
          type="monotone"
          dataKey="value"
          name="Forecast"
          stroke="#f97316"  
          strokeWidth={2}
          dot={{ r: 3 }}
          activeDot={{ r: 6 }}
        />
        {/* Upper */}
        <Line
          type="monotone"
          dataKey="upper"
          name="Upper Bound"
          stroke="#3b82f6" 
          strokeWidth={1.5}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
