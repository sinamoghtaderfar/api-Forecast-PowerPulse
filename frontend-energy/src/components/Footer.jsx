import { Link } from "react-router-dom";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

export default function Footer() {
  return (
    <Box sx={{ backgroundColor: "#000000", color: "#ffffff", py: 4, textAlign: "center" }}>
      <Typography>© 2026 PowerPulse. All rights reserved.</Typography>
      <Box sx={{ mt: 2 }}>
        <Link to="/about" style={{ margin: "0 10px", color: "#f97316" }}>About</Link>
        <Link to="/contact" style={{ margin: "0 10px", color: "#f97316" }}>Contact</Link>
        <a href="https://github.com/" target="_blank" rel="noreferrer" style={{ margin: "0 10px", color: "#f97316" }}>GitHub</a>
      </Box>
    </Box>
  );
}
