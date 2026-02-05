
import { DataGrid } from "@mui/x-data-grid";

export default function ProphetTable({ data }) {
  // data = [{ date: 2026, value: 504, lower: 483, upper: 523 }, ...]
const rows = data.map((item, index) => ({
  id: index,
  date: item.date || item.year || '—',
  value: item.value || item.forecast || null,
  lower: item.lower   || null,
  upper: item.upper   || null
}));
  const columns = [
    { field: "date", headerName: "Year", width: 120 },
    { field: "value", headerName: "Forecast", width: 150 },
    { field: "lower", headerName: "Lower Bound", width: 150 },
    { field: "upper", headerName: "Upper Bound", width: 150 }
  ];

  return (
    <div style={{ height: 300, width: "100%" }}>
      <DataGrid
        rows={rows}
        columns={columns}
        pageSize={10}                     
        rowsPerPageOptions={[5, 10, 25]}  
        pagination                        
        // checkboxSelection                 
        disableSelectionOnClick          
      />
    </div>
  );
}
