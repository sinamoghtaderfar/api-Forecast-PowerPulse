// src/components/RandomForestTable.jsx
import * as React from "react";
import { DataGrid } from "@mui/x-data-grid";

export default function RandomForestTable({ data }) {
  const rows = data.map((item, index) => ({
    id: index,
    date: item.year,
    value: item.forecast,
    lower: item.lower,
    upper: item.upper
  }));

  const columns = [
    { field: "date", headerName: "Year", width: 120 },
    { field: "value", headerName: "Forecast", width: 150 },
    { field: "lower", headerName: "Lower Bound", width: 150 },
    { field: "upper", headerName: "Upper Bound", width: 150 }
  ];

  return (
    <div style={{ width: "100%" }}>
      <DataGrid
        autoHeight
        rows={rows}
        columns={columns}
        pageSize={10}
        rowsPerPageOptions={[25, 50, 100]}
        pagination
        disableSelectionOnClick
      />
    </div>
  );
}
