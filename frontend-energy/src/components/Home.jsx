import { Link } from "react-router-dom";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Lottie from "lottie-react";
import electricityAnimation from "../assets/Japa.json";

const features = [
  { title: "Accurate Predictions", desc: "Machine learning predicts your energy consumption precisely." },
  { title: "Real-Time Data", desc: "Monitor your electricity usage live." },
  { title: "Easy to Use", desc: "Simple and intuitive interface for everyone." },
];

const steps = [
  { title: "Collect Data", desc: "Gather electricity consumption data.", icon: "📊" },
  { title: "Analyze with ML", desc: "Run machine learning algorithms.", icon: "🤖" },
  { title: "Predict Consumption", desc: "Get forecasts instantly.", icon: "⚡" },
];

export default function HomePage() {
  return (
    <div>
      {/* Hero Section */}
      <Box
        sx={{
          display: "flex",
          flexDirection: { xs: "column", md: "row" },
          alignItems: "center",
          justifyContent: "space-around",
          minHeight: "80vh",
          background: "linear-gradient(to right, #0f172a, #3b82f6)",
          color: "#ffffff",
          textAlign: "center",
          px: 2,
          gap: { xs: 2, md: 6 }
        }}
      >
        <Box sx={{ maxWidth: 500 }}>
          <Typography variant="h2" sx={{ mb: 2 }}>PowerPulse</Typography>
          <Typography variant="h5" sx={{ mb: 4 }}>
            Predicting Your Electricity Consumption with Machine Learning
          </Typography>
          <Button
            variant="contained"
            sx={{ backgroundColor: "#ad1f23", "&:hover": { backgroundColor: "#c72a2f" } }}
            component={Link}
            to="/forecast"
          >
            Start Forecasting
          </Button>
        </Box>

        <Box sx={{ width: 300, height: 300 , ml: { xs: 0, md: 8 }}}>
          <Lottie animationData={electricityAnimation} loop={true} />
        </Box>
      </Box>

      {/* Features */}
      <Box sx={{ display: "flex", justifyContent: "space-around", flexWrap: "wrap", py: 8, backgroundColor: "#f3f4f6" }}>
        {features.map((f) => (
          <Box key={f.title} sx={{ width: 280, m: 2, p: 4, borderRadius: 2, boxShadow: 3, backgroundColor: "#ffffff", textAlign: "center" }}>
            <Typography variant="h6" sx={{ mb: 2 }}>{f.title}</Typography>
            <Typography>{f.desc}</Typography>
          </Box>
        ))}
      </Box>

      {/* How It Works */}
      <Box sx={{ py: 8, textAlign: "center" }}>
        <Typography variant="h4" sx={{ mb: 4 }}>How It Works</Typography>
        <Box sx={{ display: "flex", justifyContent: "center", flexWrap: "wrap", gap: 4 }}>
          {steps.map((s) => (
            <Box key={s.title} sx={{ width: 220, p: 2, borderRadius: 2, backgroundColor: "#e0f2fe" }}>
              <Typography variant="h2">{s.icon}</Typography>
              <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>{s.title}</Typography>
              <Typography>{s.desc}</Typography>
            </Box>
          ))}
        </Box>
      </Box>

      {/* CTA Section */}
      <Box sx={{ textAlign: "center", py: 6, backgroundColor: "#0f172a" }}>
        <Button
          variant="contained"
          sx={{ backgroundColor: "#22c55e", color: "#ffffff", "&:hover": { backgroundColor: "#16a34a" }, fontSize: "1.3rem", px: 5, py: 2 }}
          component={Link}
          to="/forecast"
        >
          Go to Forecast
        </Button>
      </Box>
    </div>
  );
}
