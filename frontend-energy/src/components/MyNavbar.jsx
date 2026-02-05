import { useState } from "react";
import { Link as RouterLink } from "react-router-dom";
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Box,
  Divider,
} from "@mui/material";
import {
  Menu as MenuIcon,
  ShowChart as ChartIcon,
  ElectricBolt as BoltIcon,
  Timeline as TimelineIcon,
  ContactMail as ContactIcon,
  Brightness4 as DarkIcon,
} from "@mui/icons-material";

const menuItems = [
  { text: "Prophet Forecast", path: "/forecastprophet", icon: <ChartIcon /> },
  { text: "XGBoost Forecast", path: "/forecastxGBoost", icon: <TimelineIcon /> },
  { text: "Random Forest Forecast", path: "/forecastRandom", icon: <BoltIcon /> },
  { text: "Contact Us", path: "/contact", icon: <ContactIcon /> },
];

export default function Navbar() {
  const [open, setOpen] = useState(false);

  const toggleDrawer = (state) => () => setOpen(state);

  return (
    <>
      {/* Navbar اصلی - مشکی با گرادیانت و افکت نئونی */}
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          background: "linear-gradient(90deg, #000000, #1a0033, #000000)",
          borderBottom: "1px solid rgba(138, 43, 226, 0.3)",
          backdropFilter: "blur(10px)",
          boxShadow: "0 4px 30px rgba(138, 43, 226, 0.2)",
        }}
      >
        <Toolbar
          sx={{
            justifyContent: "space-between",
            px: { xs: 2, md: 4 },
            minHeight: 70,
          }}
        >
          {/* لوگو با گرادیانت نئونی */}
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <BoltIcon
              sx={{
                fontSize: 32,
                color: "#9f7aea",
                animation: "pulse 2s infinite",
                "@keyframes pulse": {
                  "0%, 100%": { opacity: 1 },
                  "50%": { opacity: 0.6 },
                },
              }}
            />
            <Typography
              variant="h6"
              fontWeight="bold"
              sx={{
                background: "linear-gradient(90deg, #c084fc, #a78bfa, #7c3aed)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                letterSpacing: 1.5,
              }}
            >
              PowerPulse
            </Typography>
          </Box>

          {/* منوی دسکتاپ - آیکون‌ها با hover نئونی */}
          <Box sx={{ display: { xs: "none", md: "flex" }, gap: 3 }}>
            {menuItems.map((item) => (
              <IconButton
                key={item.text}
                component={RouterLink}
                to={item.path}
                sx={{
                  color: "grey.400",
                  "&:hover": {
                    color: "#9f7aea",
                    transform: "translateY(-2px)",
                    boxShadow: "0 0 15px rgba(159, 122, 234, 0.6)",
                  },
                  transition: "all 0.3s ease",
                  borderRadius: 2,
                  p: 1,
                }}
              >
                {item.icon}
                <Typography variant="body2" sx={{ ml: 1 }}>
                  {item.text}
                </Typography>
              </IconButton>
            ))}
          </Box>

          {/* دکمه همبرگر برای موبایل */}
          <IconButton
            edge="end"
            color="inherit"
            aria-label="menu"
            onClick={toggleDrawer(true)}
            sx={{ display: { md: "none" } }}
          >
            <MenuIcon sx={{ color: "#9f7aea" }} />
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Drawer سمت چپ برای موبایل */}
      <Drawer
        anchor="left"
        open={open}
        onClose={toggleDrawer(false)}
        PaperProps={{
          sx: {
            width: 280,
            bgcolor: "#0a0a0a",
            borderRight: "1px solid #1e0033",
            backgroundImage: "linear-gradient(rgba(138,43,226,0.05), transparent)",
          },
        }}
      >
        <Box sx={{ p: 3 }}>
          <Box sx={{ display: "flex", alignItems: "center", mb: 4 }}>
            <BoltIcon sx={{ fontSize: 40, color: "#9f7aea", mr: 2 }} />
            <Typography
              variant="h5"
              sx={{
                fontWeight: "bold",
                background: "linear-gradient(90deg, #c084fc, #a78bfa)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              PowerPulse
            </Typography>
          </Box>

          <Divider sx={{ bgcolor: "rgba(159, 122, 234, 0.2)", mb: 2 }} />

          <List>
            {menuItems.map((item) => (
              <ListItem key={item.text} disablePadding>
                <ListItemButton
                  component={RouterLink}
                  to={item.path}
                  onClick={toggleDrawer(false)}
                  sx={{
                    borderRadius: 2,
                    mb: 1,
                    color: "#ffffff", // متن روشن
                    "&:hover": {
                      bgcolor: "rgba(159, 122, 234, 0.15)",
                      color: "#9f7aea",
                    },
                  }}
                >
                  <ListItemIcon sx={{ color: "#ffffff" }}>{item.icon}</ListItemIcon>
                  <ListItemText primary={item.text} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>

        </Box>
      </Drawer>

      {/* برای جلوگیری از هم‌پوشانی محتوا با navbar ثابت */}
      <Box sx={{ height: 70 }} />
    </>
  );
}