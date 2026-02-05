import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toolbar } from '@mui/material';

import MyNavbar from "./components/MyNavbar";
import Homepage from "./pages/Home";
import ForecastProphet from "./pages/ForecastProphet";
import ForecastXGBoost from "./pages/ForecastXGBoost";
import ForecastRandom from "./pages/ForecastRandom";
import Footer from "./components/Footer";

function App() {
  return (
    <BrowserRouter>
      <div className="App">
        <MyNavbar />

        <div className="AppMain">
          <Routes>
            <Route path="/" element={<Homepage />} />
            <Route path="/ForecastProphet" element={<ForecastProphet />} />
            <Route path="/forecastxGBoost" element={<ForecastXGBoost />} />
            <Route path="/forecastRandom" element={<ForecastRandom />} />
          </Routes>
        </div>

        <Footer />
      </div>
    </BrowserRouter>

  );
}

export default App;
