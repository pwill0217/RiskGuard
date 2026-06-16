from data_loader import load_data
from preprocessing import preprocess_data

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# This function loads the data, preprocesses it, trains a random forest classifier, and then creates a DataFrame to display the feature importance of each feature in the model. The features are sorted by their importance in descending order, and the top 15 features are printed to the console.
def show_feature_importance():
    df = load_data()

    x_train, x_test, y_train, y_test = preprocess_data(df)

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    importance_df = pd.DataFrame({
        "Feature": x_train.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    print(importance_df.head(15))


if __name__ == "__main__":
    show_feature_importance()

    