@echo off
REM ==============================
REM Frontend React Project Structure
REM ==============================

mkdir frontend
mkdir frontend\src
mkdir frontend\src\api
mkdir frontend\src\components
mkdir frontend\src\pages
mkdir frontend\src\hooks
mkdir frontend\src\context
mkdir frontend\src\utils
mkdir frontend\public

REM ==============================
REM Creating placeholder files
REM ==============================

echo.> frontend\src\App.jsx
echo.> frontend\src\index.jsx
echo.> frontend\src\api\energyApi.js
echo.> frontend\src\components\ChartCard.jsx
echo.> frontend\src\components\DataTable.jsx
echo.> frontend\src\pages\Dashboard.jsx
echo.> frontend\src\pages\Forecast.jsx
echo.> frontend\src\hooks\useEnergyData.js
echo.> frontend\src\context\EnergyContext.jsx
echo.> frontend\src\utils\helpers.js
echo.> frontend\public\index.html
echo.> frontend\package.json
echo.> frontend\vite.config.js

echo Frontend React project structure created successfully!
pause
