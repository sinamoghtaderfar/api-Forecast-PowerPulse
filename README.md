# Forecast PowerPulse API

This repository contains the backend API for the Forecast PowerPulse project. The API provides endpoints for electricity consumption forecasting in Germany and serves as the backend for the web frontend application in the [frontend-PowerPulse](https://github.com/sinamoghtaderfar/frontend-PowerPulse) repository.

A full demonstration of the system is available in this YouTube video:  
https://youtu.be/oULub0Q4xnE

## Overview

Forecast PowerPulse predicts electricity consumption in Germany using **40 years of historical data (1985–2025)**.  
The system employs three complementary machine learning models:

- **Prophet** – Captures long-term trends and seasonality in time series data  
- **Random Forest** – Handles complex non-linear relationships between features  
- **XGBoost** – Provides high accuracy and robustness for structured tabular data  

**Performance Note:**  
The first API request triggers model training and takes approximately **4–7 seconds**. All subsequent requests are served from cache in **under 500 ms** using a smart data hashing mechanism.

## Data

The system uses electricity consumption and related economic data for Germany from **1985 to 2025**, sourced from the  
[Energy Dataset](https://github.com/owid/energy-data).

### Input Features

| Feature            | Description                                   | Source        |
|--------------------|-----------------------------------------------|---------------|
| GDP                | Gross Domestic Product                         | Historical    |
| Population         | Total population                               | Historical    |
| Renewables Share   | Percentage of renewable energy usage           | Historical    |
| GDP per capita     | Economic output per person                     | Calculated    |
| Energy Intensity   | Energy consumption relative to GDP             | Calculated    |

## Output

- **10-year electricity consumption forecast**  
- Annual predictions with **95% confidence intervals**  
- Lower and upper bounds for each forecasted year

## Installation

1. Clone the repository:
git clone https://github.com/sinamoghtaderfar/api-Forecast-PowerPulse.git

2. Change into the project directory:
cd api-Forecast-PowerPulse

3. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate      # Linux or mac
venv\Scripts\activate         # Windows

4. Install dependencies:
pip install -r requirements.txt

Running the API

Start the API server locally:
python run.py

By default, the API will be available at http://localhost:5000.

Note on First Request

The first request for a forecast will take longer, as the models need to train on historical data. After the initial training, the models use a data hashing mechanism to provide faster responses for subsequent requests.

Integrating with the Frontend

Ensure the frontend is configured to use the backend API URL. For local development:
http://localhost:5000

Adjust for production deployment accordingly.

Development

Install dependencies.

Start the API server (python run.py).

Start the frontend (npm install and npm run dev).

Test features with live backend data.

