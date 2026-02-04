// src/App.jsx
import { BrowserRouter} from "react-router-dom";
import { Toolbar } from '@mui/material';


import MyNavbar from "./components/MyNavbar";
import Homepage from "./components/Home";
import Footer from "./components/Footer";


function App() {
  return (
    <BrowserRouter>
      <MyNavbar/>
      <Toolbar/>
      <Homepage/>
      <Footer/>
    </BrowserRouter>
  );
}

export default App;