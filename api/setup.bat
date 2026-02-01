@echo off

mkdir app
mkdir app\data
mkdir app\controllers
mkdir app\models
mkdir app\services
mkdir app\schemas
mkdir app\utils

echo.> app\__init__.py
echo.> app\data\germany-energy-clean.csv
echo.> app\controllers\energy_controller.py
echo.> app\models\energy_model.py
echo.> app\services\forecast_service.py
echo.> app\schemas\energy_schema.py
echo.> app\utils\helpers.py
echo.> run.py
echo.> requirements.txt

echo Project structure created successfully!
pause
