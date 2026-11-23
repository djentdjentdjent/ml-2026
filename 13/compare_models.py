import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, r2_score


# Load the dataset
df = pd.read_csv("afghanistan_temperature_ml_dataset.csv")


# Use only simple user inputs
features = [
    "city",
    "year",
    "month"
]

target = "temperature_c"

X = df[features]
y = df[target]


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Encode the city column
preprocessor = ColumnTransformer(
    transformers=[
        (
            "city_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            ["city"]
        )
    ],
    remainder="passthrough"
)


# Define models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


results = []


# Train and evaluate each model
for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "R2": round(r2, 3)
    })


# Display results
results_df = pd.DataFrame(results)

print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))