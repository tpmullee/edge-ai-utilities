import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_linear_regression(X, y):
    """
    Trains a linear regression model on the provided features and target.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    logging.info("Linear Regression training complete.")
    return model, mse

def train_random_forest(X, y, param_grid=None):
    """
    Trains a Random Forest model using GridSearchCV for hyperparameter tuning.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20]
        }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    logging.info("Random Forest training complete with best parameters: %s", grid_search.best_params_)
    return best_model, mse

def save_model(model, filename):
    """
    Saves the trained model to a file using joblib.
    """
    try:
        joblib.dump(model, filename)
        logging.info(f"Model saved to '{filename}'.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def main():
    parser = argparse.ArgumentParser(description="Model Training Utility")
    parser.add_argument("--model", type=str, choices=["linear", "rf"], default="linear",
                        help="Model type: 'linear' for Linear Regression, 'rf' for Random Forest")
    parser.add_argument("--save", type=str, default="model.pkl", help="Filename to save the trained model")
    args = parser.parse_args()

    # Example dataset
    data = {
        'feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'target': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    }
    df = pd.DataFrame(data)
    X = df[['feature']]
    y = df['target']

    if args.model == "linear":
        model, mse = train_linear_regression(X, y)
    else:
        model, mse = train_random_forest(X, y)

    logging.info("Model training completed with MSE: %f", mse)
    print(f"Trained model MSE: {mse}")
    
    save_model(model, args.save)

if __name__ == "__main__":
    main()
