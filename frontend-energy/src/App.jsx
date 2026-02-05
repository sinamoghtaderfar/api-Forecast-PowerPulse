import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toolbar } from '@mui/material';

import MyNavbar from "./components/MyNavbar";
import Homepage from "./pages/Home";
import Forecast from "./pages/Forecast";
import Footer from "./components/Footer";

function App() {
  return (
    <BrowserRouter>
      <div className="App">
        <MyNavbar />
        <Toolbar />

        <div className="AppMain">
          <Routes>
            <Route path="/" element={<Homepage />} />
            <Route path="/forecast" element={<Forecast />} />
          </Routes>
        </div>

        <Footer />
      </div>
    </BrowserRouter>

  );
}

export default App;
