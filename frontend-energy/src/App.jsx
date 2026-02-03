// src/App.jsx
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Forecast from "./pages/Forecast";

function App() {
  return (
    <BrowserRouter>
      <div style={{ padding: "20px", textAlign: "center" }}>
        <h1>Energy</h1>
        <Link to="/forecast" style={{ fontSize: "1.4rem", color: "#0066cc" }}>
          Click
        </Link>
      </div>

      <Routes>
        <Route path="/" element={<div>Click</div>} />
        <Route path="/forecast" element={<Forecast />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;