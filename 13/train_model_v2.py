import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load the dataset
df = pd.read_csv("afghanistan_temperature_ml_dataset.csv")


# Select input features and target
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


# Create the Gradient Boosting model
model = GradientBoostingRegressor(
    random_state=42
)


# Build the pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train the model
pipeline.fit(X_train, y_train)


# Evaluate the model
predictions = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("=== MODEL V2 RESULTS ===")
print("Mean Absolute Error:", round(mae, 3))
print("R2 Score:", round(r2, 3))


# Save the trained model
joblib.dump(
    pipeline,
    "temperature_model_v2.pkl"
)

print("\nModel saved as temperature_model_v2.pkl")