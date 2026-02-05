import { Link as RouterLink } from "react-router-dom";
import {
  Box,
  Typography,
  Button,
  Container,
  Grid,
  Card,
  CardContent,
  Divider,
} from "@mui/material";
import Lottie from "lottie-react";
import electricityAnimation from "../assets/Japa.json";

export default function HomePage() {
  return (
    <Box sx={{ bgcolor: "#000000", color: "grey.300", minHeight: "100vh" }}>
      {/* Hero Section */}
      <Box
        sx={{
          position: "relative",
          minHeight: "90vh",
          display: "flex",
          alignItems: "center",
          background:
            "linear-gradient(135deg, #000000 0%, #1a0033 50%, #000000 100%)",
          borderBottom: "1px solid rgba(138, 43, 226, 0.3)",
          overflow: "hidden",
        }}
      >
        {/* افکت پس‌زمینه */}
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            background:
              "radial-gradient(circle at 30% 70%, rgba(138,43,226,0.12), transparent 60%)",
            pointerEvents: "none",
          }}
        />

        <Container maxWidth="lg">
          <Grid container spacing={6} alignItems="center">
            {/* متن و دکمه‌ها */}
            <Grid item xs={12} md={6}>
              <Typography
                variant="h1"
                fontWeight="bold"
                sx={{
                  fontSize: { xs: "2.8rem", md: "4.5rem" },
                  background:
                    "linear-gradient(90deg, #c084fc, #a78bfa, #7c3aed)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  mb: 3,
                  letterSpacing: "-0.5px",
                }}
              >
                PowerPulse
              </Typography>

              <Typography
                variant="h5"
                sx={{
                  mb: 5,
                  maxWidth: 520,
                  color: "grey.300",
                  lineHeight: 1.6,
                }}
              >
                Advanced electricity consumption forecasting using cutting-edge
                machine learning models.
              </Typography>

              <Box sx={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
                <Button
                  variant="contained"
                  size="large"
                  component={RouterLink}
                  to="/forecastprophet"
                  sx={{
                    background: "linear-gradient(90deg, #7c3aed, #a78bfa)",
                    color: "white",
                    px: 5,
                    py: 1.5,
                    fontSize: "1.1rem",
                    fontWeight: "bold",
                    borderRadius: 2,
                    boxShadow: "0 0 20px rgba(124, 58, 237, 0.5)",
                    "&:hover": {
                      boxShadow: "0 0 35px rgba(124, 58, 237, 0.7)",
                      transform: "translateY(-3px)",
                    },
                    transition: "all 0.3s ease",
                  }}
                >
                  Try Prophet Model
                </Button>

                <Button
                  variant="outlined"
                  size="large"
                  component={RouterLink}
                  to="/forecastxGBoost"
                  sx={{
                    borderColor: "#9f7aea",
                    color: "#9f7aea",
                    px: 5,
                    py: 1.5,
                    fontSize: "1.1rem",
                    borderRadius: 2,
                    "&:hover": {
                      borderColor: "#c084fc",
                      color: "#c084fc",
                      bgcolor: "rgba(159, 122, 234, 0.08)",
                    },
                  }}
                >
                  Explore XGBoost
                </Button>
              </Box>

              <Typography variant="body2" sx={{ mt: 4, color: "grey.500" }}>
                Also available: Random Forest • Prophet • XGBoost
              </Typography>
            </Grid>

            {/* انیمیشن وسط‌چین */}
            <Grid item xs={12} md={6}>
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  width: "100%",
                  height: { xs: "300px", sm: "350px", md: "400px" },
                }}
              >
                <Lottie
                  animationData={electricityAnimation}
                  loop
                  autoplay
                  style={{
                    width: "100%",
                    height: "100%",
                    maxWidth: "400px",
                    maxHeight: "100%",
                    filter: "drop-shadow(0 0 40px rgba(138, 43, 226, 0.4))",
                  }}
                />
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* بخش مدل‌ها */}
      <Container maxWidth="lg" sx={{ py: 12 }}>
        <Typography
          variant="h3"
          align="center"
          sx={{
            mb: 10,
            fontWeight: "bold",
            background: "linear-gradient(90deg, #c084fc, #a78bfa)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Choose Your Forecasting Model
        </Typography>

        <Box sx={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {/* کارت‌ها */}
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <Card
                sx={{
                  height: "100%",
                  bgcolor: "#0a0a0a",
                  border: "1px solid rgba(138,43,226,0.25)",
                  borderRadius: 3,
                  transition: "all 0.4s ease",
                  "&:hover": {
                    transform: "translateY(-12px)",
                    boxShadow: "0 20px 40px rgba(138,43,226,0.3)",
                    borderColor: "#c084fc",
                  },
                }}
              >
                <CardContent sx={{ p: 5, textAlign: "center" }}>
                  <Typography
                    variant="h5"
                    fontWeight="bold"
                    color="#c084fc"
                    gutterBottom
                  >
                    Prophet Model
                  </Typography>
                  <Typography
                    variant="body1"
                    color="grey.400"
                    sx={{ mb: 3, minHeight: 100 }}
                  >
                    Time-series forecasting with strong support for seasonality,
                    trends and holidays.
                  </Typography>
                  <Button
                    variant="outlined"
                    component={RouterLink}
                    to="/forecastprophet"
                    sx={{
                      borderColor: "#c084fc",
                      color: "#c084fc",
                      "&:hover": { bgcolor: "rgba(192,132,252,0.1)" },
                    }}
                  >
                    Try Prophet
                  </Button>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6} sx={{ display: { xs: "none", md: "block" } }} />
          </Grid>

          <Grid
            container
            spacing={4}
            alignItems="center"
            direction={{ xs: "column", md: "row-reverse" }}
          >
            <Grid item xs={12} md={6} sx={{ display: { xs: "none", md: "block" } }} />

            <Grid item xs={12} md={6}>
              <Card
                sx={{
                  height: "100%",
                  bgcolor: "#0a0a0a",
                  border: "1px solid rgba(138,43,226,0.25)",
                  borderRadius: 3,
                  transition: "all 0.4s ease",
                  "&:hover": {
                    transform: "translateY(-12px)",
                    boxShadow: "0 20px 40px rgba(138,43,226,0.3)",
                    borderColor: "#a78bfa",
                  },
                }}
              >
                <CardContent sx={{ p: 5, textAlign: "center" }}>
                  <Typography
                    variant="h5"
                    fontWeight="bold"
                    color="#a78bfa"
                    gutterBottom
                  >
                    XGBoost Model
                  </Typography>
                  <Typography
                    variant="body1"
                    color="grey.400"
                    sx={{ mb: 3, minHeight: 100 }}
                  >
                    High-performance gradient boosting – excellent for tabular data
                    and complex patterns.
                  </Typography>
                  <Button
                    variant="outlined"
                    component={RouterLink}
                    to="/forecastxGBoost"
                    sx={{
                      borderColor: "#a78bfa",
                      color: "#a78bfa",
                      "&:hover": { bgcolor: "rgba(167,139,250,0.1)" },
                    }}
                  >
                    Try XGBoost
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <Card
                sx={{
                  height: "100%",
                  bgcolor: "#0a0a0a",
                  border: "1px solid rgba(138,43,226,0.25)",
                  borderRadius: 3,
                  transition: "all 0.4s ease",
                  "&:hover": {
                    transform: "translateY(-12px)",
                    boxShadow: "0 20px 40px rgba(138,43,226,0.3)",
                    borderColor: "#7c3aed",
                  },
                }}
              >
                <CardContent sx={{ p: 5, textAlign: "center" }}>
                  <Typography
                    variant="h5"
                    fontWeight="bold"
                    color="#7c3aed"
                    gutterBottom
                  >
                    Random Forest
                  </Typography>
                  <Typography
                    variant="body1"
                    color="grey.400"
                    sx={{ mb: 3, minHeight: 100 }}
                  >
                    Robust ensemble method – good balance between accuracy and
                    interpretability.
                  </Typography>
                  <Button
                    variant="outlined"
                    component={RouterLink}
                    to="/forecastRandom"
                    sx={{
                      borderColor: "#7c3aed",
                      color: "#7c3aed",
                      "&:hover": { bgcolor: "rgba(124,58,237,0.1)" },
                    }}
                  >
                    Try Random Forest
                  </Button>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6} sx={{ display: { xs: "none", md: "block" } }} />
          </Grid>
        </Box>
      </Container>

      <Divider sx={{ borderColor: "rgba(138,43,226,0.2)", my: 6 }} />

      <Container maxWidth="lg" sx={{ pb: 12, textAlign: "center" }}>
        <Typography variant="h5" color="grey.400">
          Built with advanced ML • Real-time insights • Future-ready forecasting
        </Typography>
      </Container>
    </Box>
  );
}
