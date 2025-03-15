import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  

MODEL_FOLDER = "api/models"
os.makedirs(MODEL_FOLDER, exist_ok=True)  

class ModelTrainer:
    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        
        predictions = model.predict(X_test)
        performance = r2_score(y_test, predictions)  

        
        model_path = os.path.join(MODEL_FOLDER, "model.pkl")
        joblib.dump(model, model_path)

        
        if os.path.exists(model_path):
            print("Model saved successfully at:", model_path)
        else:
            print("Model was NOT saved!")

        return performance, model
