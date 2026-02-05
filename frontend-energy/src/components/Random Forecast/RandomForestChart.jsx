// src/components/RandomForestChart.jsx
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

export default function RandomForestChart({ data }) {
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
        <Line type="monotone" dataKey="lower" name="Lower Bound" stroke="#ef4444" dot={false} />
        <Line type="monotone" dataKey="value" name="Random Forest Forecast" stroke="#f97316" dot={{ r: 3 }} activeDot={{ r: 6 }} />
        <Line type="monotone" dataKey="upper" name="Upper Bound" stroke="#3b82f6" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
