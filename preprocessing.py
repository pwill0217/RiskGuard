import pandas as pd
from sklearn.model_selection import train_test_split # this is used to split the data into training and testing sets
# LabelEncoder is used to convert categorical variables into numerical format, which is necessary for machine learning algorithms that require numerical input. It assigns a unique integer to each category in the categorical variable, allowing the model to process the data effectively.
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    # Drop rows with missing target values
    df = df.dropna(subset=["Status"])

    # Fill numeric missing values with the median
    number_columns = df.select_dtypes(include=["float64", "int64"]).columns

    for col in number_columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with the mode
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical variables only
    encoder = LabelEncoder()

    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])

    # Features and target
    columns_to_drop = [
    "ID",   
    "Interest_rate_spread",
    "Upfront_charges",
    "rate_of_interest"
]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    x = df.drop("Status", axis=1)
    y = df["Status"]

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    return x_train, x_test, y_train, y_test