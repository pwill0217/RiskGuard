from data_loader import load_data
from preprocessing import preprocess_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt # this is used to create visualizations of the data and the model's performance. It provides a wide range of plotting functions that allow us to create various types of charts and graphs, such as histograms, scatter plots, and heatmaps. In this code, we use matplotlib to visualize the income distribution, the relationship between income and loan amount, the correlation between features, and the confusion matrix of the random forest model.
import pandas as pd


def generate_visualizations():

    df = load_data()

    # HISTOGRAM
    df["income"].hist(bins=30)
    plt.xlim(0, 100000)

    plt.title("Income Distribution")
    plt.xlabel("Income")
    plt.ylabel("Frequency")
    plt.show()

    # SCATTER PLOT
    plt.scatter(df["income"], df["loan_amount"])

    plt.title("Income vs Loan Amount")
    plt.xlabel("Income")
    plt.ylabel("Loan Amount")
    plt.show()

    # CORRELATION HEATMAP
    columns_to_drop = [
    "ID",
    "rate_of_interest",
    "Interest_rate_spread",
    "Upfront_charges"
]

    df = df.drop(columns=columns_to_drop, errors="ignore")
    correlation = df.select_dtypes(include=["float64", "int64"]).corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(correlation, cmap="coolwarm", aspect="auto")
    plt.colorbar()

    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.columns)), correlation.columns)

    plt.title("Correlation Heatmap")
    plt.show()

    # CONFUSION MATRIX
    x_train, x_test, y_train, y_test = preprocess_data(df)

    model = RandomForestClassifier(random_state=42)

    model.fit(x_train, y_train)

    ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test
    )

    plt.title("Random Forest Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    generate_visualizations()