from data_loader import load_data # this is used to load the data from the specified path and return a pandas DataFrame. It also checks if the file exists before attempting to load it, and raises a FileNotFoundError if it does not.
from preprocessing import preprocess_data # this is used to preprocess the data by handling missing values, encoding categorical variables, and splitting the data into training and testing sets. It ensures that the data is in a suitable format for machine learning algorithms.

from sklearn.linear_model import LogisticRegression # this is used to create a logistic regression model, which is a commonly used algorithm for binary classification tasks. It models the relationship between the features and the target variable by estimating the probabilities of the different classes.
from sklearn.ensemble import RandomForestClassifier # this is used to create a random forest classifier, which is an ensemble learning method that combines multiple decision trees to improve the accuracy and robustness of the model. It works by training multiple decision trees on different subsets of the data and then aggregating their predictions to make a final prediction.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # these are used to evaluate the performance of the models. Accuracy measures the overall correctness of the model, precision measures the proportion of true positive predictions among all positive predictions, recall measures the proportion of true positive predictions among all actual positives, F1 score is the harmonic mean of precision and recall, and confusion matrix provides a summary of the prediction results by showing the counts of true positives, true negatives, false positives, and false negatives.
import joblib # this is used to save the trained model to a file, allowing us to reuse the model later without having to retrain it. It provides a simple way to serialize and deserialize Python objects, making it easy to save and load machine learning models.
from pathlib import Path # this is used to handle file paths in a platform-independent way. It allows us to construct file paths that work across different operating systems, making our code more portable and easier to maintain.


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "loan_model.joblib"


def train_models(): # this is the main function that orchestrates the entire process of loading the data, preprocessing it, training the models, evaluating their performance, and saving the best model. It first loads the data using the load_data function, then preprocesses it using the preprocess_data function to get the training and testing sets. Next, it trains both a logistic regression model and a random forest classifier on the training data. After training, it evaluates each model's performance on the test set using various metrics and prints the results. Finally, it saves the best-performing model (in this case, the random forest model) to a file for future use.
    df = load_data()

    x_train, x_test, y_train, y_test = preprocess_data(df)

    logistic_model = LogisticRegression(max_iter=1000) # this is used to create a logistic regression model with a maximum number of iterations set to 1000. This is done to ensure that the model has enough iterations to converge during training, especially if the dataset is large or complex. By increasing the max_iter parameter, we can help prevent issues with convergence and improve the performance of the logistic regression model.
    logistic_model.fit(x_train, y_train)

    random_forest_model = RandomForestClassifier(random_state=42) # this is used to create a random forest classifier with a specified random state for reproducibility. The random_state parameter ensures that the results of the model training are consistent across different runs, allowing us to compare the performance of the model reliably. By setting random_state to a fixed value (in this case, 42), we can ensure that the same random numbers are generated during the training process, which can be helpful for debugging and comparing results.
    random_forest_model.fit(x_train, y_train) # this is used to train the random forest model on the training data (x_train and y_train). The fit method is called to fit the model to the training data, allowing it to learn the patterns and relationships in the data. After calling fit, the random forest model will be trained and ready to make predictions on new data.

    models = {
        "Logistic Regression": logistic_model, # this is used to create a dictionary that maps the names of the models to their corresponding trained model objects. This allows us to easily iterate over the models and evaluate their performance in a consistent way. By using a dictionary, we can store multiple models and their names together, making it easier to manage and compare their results.
        "Random Forest": random_forest_model
    }
 # this is used to iterate over the models in the dictionary and evaluate their performance on the test set. For each model, it makes predictions using the predict method and then calculates various performance metrics (accuracy, precision, recall, F1 score) using the corresponding functions from sklearn.metrics. It also prints the confusion matrix for each model to provide a summary of the prediction results. This allows us to compare the performance of the different models and determine which one performs better on the test data.
    for name, model in models.items():
        predictions = model.predict(x_test)

        print(f"\n{name} Results")
        print("Accuracy:", accuracy_score(y_test, predictions))
        print("Precision:", precision_score(y_test, predictions))
        print("Recall:", recall_score(y_test, predictions))
        print("F1 Score:", f1_score(y_test, predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

    joblib.dump(random_forest_model, MODEL_PATH)
    print(f"\nBest model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_models()