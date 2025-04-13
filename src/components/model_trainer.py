import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

MODEL_FOLDER = "api/models"
os.makedirs(MODEL_FOLDER, exist_ok=True)
7
class ModelTrainer:
    def __init__(self):
        self.models = {
            "regression": {
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                "GradientBoosting": GradientBoostingRegressor(random_state=42),
                "LinearRegression": LinearRegression(),
                "SVR": SVR()
            },
            "classification": {
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                "GradientBoosting": GradientBoostingClassifier(random_state=42),
                "LogisticRegression": LogisticRegression(),
                "SVC": SVC()
            }
        }

    def train_best_model(self, X, y, task_type="regression"):
      
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_model = None
        best_score = float('-inf') if task_type == "regression" else 0
        best_model_name = ""

        models_to_try = self.models.get(task_type, {})
        metric = r2_score if task_type == "regression" else accuracy_score

        for model_name, model in models_to_try.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            if task_type == "classification":
                predictions = np.round(predictions)

            score = metric(y_test, predictions)

            if (task_type == "regression" and score > best_score) or (task_type == "classification" and score > best_score):
                best_score = score
                best_model = model
                best_model_name = model_name

        model_path = os.path.join(MODEL_FOLDER, "best_model.pkl")
        joblib.dump(best_model, model_path)

        print(f"Best model: {best_model_name} with score: {best_score}")
        print(f"Model saved at: {model_path}")

        return best_model_name, best_score, best_model
