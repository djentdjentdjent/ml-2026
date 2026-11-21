from flask import Flask, render_template, request
import pandas as pd
import joblib
import calendar

app = Flask(__name__)

# Load the trained V2 machine learning model
model = joblib.load("temperature_model_v2.pkl")

# Load historical climate data
data = pd.read_csv("afghanistan_temperature_ml_dataset.csv")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    selected_city = ""
    selected_year = ""
    selected_month = ""
    selected_month_name = ""

    chart_years = []
    chart_temperatures = []

    if request.method == "POST":
        try:
            selected_city = request.form["city"]
            selected_year = request.form["year"]
            selected_month = request.form["month"]

            city = selected_city
            year = int(selected_year)
            month = int(selected_month)

            # Validation
            if city == "":
                raise ValueError("Please select a city.")

            if year < 2010 or year > 2035:
                raise ValueError("Year must be between 2010 and 2035.")

            if month < 1 or month > 12:
                raise ValueError("Month must be between 1 and 12.")

            # Prepare input data for the ML model
            input_data = pd.DataFrame([
                {
                    "city": city,
                    "year": year,
                    "month": month
                }
            ])

            # Make prediction
            result = model.predict(input_data)
            prediction = round(result[0], 2)

            selected_month_name = calendar.month_name[month]

            # Prepare historical data for the chart
            historical_data = data[
                (data["city"] == city) &
                (data["month"] == month)
            ].sort_values("year")

            chart_years = historical_data["year"].tolist()

            chart_temperatures = [
                round(value, 2)
                for value in historical_data["temperature_c"].tolist()
            ]

        except ValueError as e:
            error = str(e)

        except Exception:
            error = "Something went wrong while making the prediction."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        selected_city=selected_city,
        selected_year=selected_year,
        selected_month=selected_month,
        selected_month_name=selected_month_name,
        chart_years=chart_years,
        chart_temperatures=chart_temperatures
    )


if __name__ == "__main__":
    app.run(debug=True)