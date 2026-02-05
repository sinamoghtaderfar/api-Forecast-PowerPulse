import React from 'react';
import {
  Box,
  Container,
  Grid,
  Typography,
  Link,
  IconButton,
  Divider,
} from '@mui/material';
import {
  GitHub,
  LinkedIn,
  Twitter,
  Instagram,
  Email,
  Phone,
  LocationOn,
  Copyright,
} from '@mui/icons-material';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <Box
      component="footer"
      sx={{
        bgcolor: '#0a0a0a',
        color: 'grey.400',
        pt: 8,
        pb: 4,
        borderTop: '1px solid #1e1e1e',
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={6}>
          {/*Abaut us*/}
          <Grid item xs={12} md={4}>
            <Typography
              variant="h5"
              fontWeight="bold"
              sx={{
                color: 'white',
                mb: 2,
                letterSpacing: 1.5,
                background: 'linear-gradient(90deg, #c084fc, #a78bfa)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              PowerPulse
            </Typography>
            <Typography variant="body2" sx={{ mb: 3, lineHeight: 1.8 }}>
              Building the future with clean code, modern design, and unstoppable energy.
            </Typography>

            {/* media*/}
            <Box sx={{ display: 'flex', gap: 1.5 }}>
              <IconButton
                component="a"
                href="https://github.com/sinamoghtaderfar"
                target="_blank"
                rel="noopener noreferrer"
                sx={{
                  color: 'grey.400',
                  '&:hover': { color: '#f0f0f0', transform: 'translateY(-3px)' },
                  transition: 'all 0.3s',
                  bgcolor: 'rgba(255,255,255,0.05)',
                }}
              >
                <GitHub fontSize="medium" />
              </IconButton>

              <IconButton
                component="a"
                href="https://linkedin.com/in/sinamoghtaderfar"
                target="_blank"
                sx={{
                  color: 'grey.400',
                  '&:hover': { color: '#0a66c2', transform: 'translateY(-3px)' },
                  transition: 'all 0.3s',
                  bgcolor: 'rgba(255,255,255,0.05)',
                }}
              >
                <LinkedIn fontSize="medium" />
              </IconButton>

              <IconButton
                component="a"
                href="https://twitter.com/yourhandle"
                target="_blank"
                sx={{
                  color: 'grey.400',
                  '&:hover': { color: '#1da1f2', transform: 'translateY(-3px)' },
                  transition: 'all 0.3s',
                  bgcolor: 'rgba(255,255,255,0.05)',
                }}
              >
                <Twitter fontSize="medium" />
              </IconButton>

              <IconButton
                component="a"
                href="https://instagram.com/yourprofile"
                target="_blank"
                sx={{
                  color: 'grey.400',
                  '&:hover': { color: '#e1306c', transform: 'translateY(-3px)' },
                  transition: 'all 0.3s',
                  bgcolor: 'rgba(255,255,255,0.05)',
                }}
              >
                <Instagram fontSize="medium" />
              </IconButton>
            </Box>
          </Grid>

          {/*  Quick Links */}
          <Grid item xs={6} sm={3} md={2}>
            <Typography variant="subtitle1" fontWeight="bold" color="white" gutterBottom>
              Quick Links
            </Typography>
            <Box component="ul" sx={{ listStyle: 'none', p: 0, m: 0 }}>
              {['Home', 'About', 'Services', 'Blog', 'Contact'].map((item) => (
                <Box component="li" key={item} sx={{ mb: 1.5 }}>
                  <Link
                    href="#"
                    color="inherit"
                    underline="hover"
                    sx={{
                      fontSize: '0.95rem',
                      '&:hover': { color: 'primary.light' },
                    }}
                  >
                    {item}
                  </Link>
                </Box>
              ))}
            </Box>
          </Grid>

          {/*  Contact Us */}
          <Grid item xs={6} sm={3} md={3}>
            <Typography variant="subtitle1" fontWeight="bold" color="white" gutterBottom>
              Contact Us
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                <LocationOn fontSize="small" color="primary" />
                <Typography variant="body2">
                  123 Future St, Tech City, 10101
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                <Email fontSize="small" color="primary" />
                <Typography variant="body2">hello@yourbrand.com</Typography>
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                <Phone fontSize="small" color="primary" />
                <Typography variant="body2">+1 (555) 123-4567</Typography>
              </Box>
            </Box>
          </Grid>

          {/*  Newsletter  */}
          <Grid item xs={12} md={3}>
            <Typography variant="subtitle1" fontWeight="bold" color="white" gutterBottom>
              Stay Updated
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Subscribe to our newsletter for the latest updates.
            </Typography>
            {/* newsletter */}
            <Box
              component="form"
              sx={{
                display: 'flex',
                flexDirection: 'column',
                gap: 1,
              }}
            >
              <input
                type="email"
                placeholder="Your email"
                style={{
                  padding: '10px 14px',
                  borderRadius: '8px',
                  border: '1px solid #333',
                  background: '#111',
                  color: 'white',
                  outline: 'none',
                }}
              />
              <button
                type="submit"
                style={{
                  padding: '10px',
                  borderRadius: '8px',
                  background: 'linear-gradient(90deg, #7c3aed, #a78bfa)',
                  color: 'white',
                  border: 'none',
                  fontWeight: 'bold',
                  cursor: 'pointer',
                }}
              >
                Subscribe
              </button>
            </Box>
          </Grid>
        </Grid>

        <Divider sx={{ my: 6, borderColor: '#1e1e1e' }} />

        {/* copy*/}
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            justifyContent: 'space-between',
            alignItems: 'center',
            textAlign: 'center',
            gap: 2,
          }}
        >
          <Typography variant="body2" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Copyright fontSize="small" /> {currentYear} YourBrand. All rights reserved.
          </Typography>

          <Typography variant="body2">
            Made with <Box component="span" sx={{ color: 'error.main' }}>♥</Box> by the dream team
          </Typography>

          <Box sx={{ display: 'flex', gap: 3 }}>
            <Link href="#" color="inherit" underline="hover" fontSize="0.9rem">
              Privacy Policy
            </Link>
            <Link href="#" color="inherit" underline="hover" fontSize="0.9rem">
              Terms of Service
            </Link>
          </Box>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;