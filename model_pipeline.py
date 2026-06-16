# this file is used to convert the raw data into a format that can be used for when the user inputs their data for the app
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_loader import load_data


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "riskguard_pipeline.joblib"


def train_pipeline():
    df = load_data()

    columns_to_drop = [
        "ID",
        "Interest_rate_spread",
        "Upfront_charges",
        "rate_of_interest"
    ]

    df = df.drop(columns=columns_to_drop, errors="ignore")

    x = df.drop("Status", axis=1)
    y = df["Status"]

    numeric_columns = x.select_dtypes(include=["int64", "float64"]).columns
    categorical_columns = x.select_dtypes(include=["object"]).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(random_state=42))
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    model_pipeline.fit(x_train, y_train)

    predictions = model_pipeline.predict(x_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(model_pipeline, MODEL_PATH)

    print(f"Pipeline saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_pipeline()