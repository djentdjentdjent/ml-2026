# Afghanistan Temperature Prediction

A machine learning web application for predicting the average monthly temperature in selected cities of Afghanistan.

## Project Goal

The goal of this project is to build a working web application that uses a machine learning model to estimate the average monthly temperature for selected cities in Afghanistan.

The user selects:

- City
- Year
- Month

The application then predicts the average monthly temperature using a trained machine learning model.

## Dataset

The project uses ERA5-Land climate data from 2010 to 2025.

The dataset contains data for the following cities:

- Kabul
- Herat
- Kandahar
- Mazar-e-Sharif
- Jalalabad
- Kunduz
- Bamyan
- Ghazni
- Farah
- Faizabad

## Machine Learning

Several regression models were compared:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The Gradient Boosting Regressor was selected as the final model.

### Model Performance

- MAE: 1.416 °C
- R² Score: 0.976

### Input Features

The final model uses:

- City
- Year
- Month

### Target

- Average monthly temperature in °C

## Technologies

- Python
- Pandas
- Scikit-learn
- Flask
- HTML
- CSS
- Chart.js

## Project Structure

```text
Afghanistan_Temperature_ML/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── afghanistan_temperature_ml_dataset.csv
├── app.py
├── compare_models.py
├── temperature_model_v2.pkl
├── train_model_v2.py
├── requirements.txt
├── .gitignore
└── README.md