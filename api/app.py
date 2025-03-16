from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import joblib
import json
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

app = Flask(__name__, template_folder="../templates")

MODEL_FOLDER = "api/models"
os.makedirs(MODEL_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html", trained=False)

@app.route("/train", methods=["POST"])
def train():
    try:
        file = request.files["file"]
        target_column = request.form["target_column"]

        df = pd.read_csv(file)
        print("Data Loaded Successfully")

        data_transformation = DataTransformation(target_column)
        X, y, preprocessor = data_transformation.preprocess_data(df)
        print("Data Preprocessed Successfully")

        y_series = pd.Series(y)  
        task_type = "classification" if y_series.nunique() <= 10 and y_series.dtype in ['int64', 'int32'] else "regression"

        
        model_trainer = ModelTrainer()
        best_model_name, best_score, best_model = model_trainer.train_best_model(X, y, task_type)
        print(f"Best Model Selected: {best_model_name} with Score: {best_score}")

        model_path = os.path.join(MODEL_FOLDER, "best_model.pkl")
        preprocessor_path = os.path.join(MODEL_FOLDER, "preprocessor.pkl")

        joblib.dump(best_model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        print(f"Best Model saved successfully at: {model_path}")

        best_model_data = {
            "best_model": best_model_name,
            "score": best_score
        }
        with open(os.path.join(MODEL_FOLDER, "best_model.json"), "w") as f:
            json.dump(best_model_data, f)

        return jsonify(best_model_data)

    except Exception as e:
        print(f"Error during training: {str(e)}")  
        return jsonify({"error": str(e)}), 500  

@app.route("/get_best_model", methods=["GET"])
def get_best_model():
    try:
        best_model_path = os.path.join(MODEL_FOLDER, "best_model.json")
        
        if not os.path.exists(best_model_path):
            return jsonify({"error": "No trained model found"}), 404

        with open(best_model_path, "r") as f:
            best_model_data = json.load(f)

        return jsonify(best_model_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    try:
        model_path = os.path.join(MODEL_FOLDER, "best_model.pkl")
        preprocessor_path = os.path.join(MODEL_FOLDER, "preprocessor.pkl")

        if not os.path.exists(model_path):
            print("Model file missing!")
            return jsonify({"error": "No trained model found"}), 400

        if not os.path.exists(preprocessor_path):
            print("Preprocessor file missing!")
            return jsonify({"error": "No trained preprocessor found"}), 400

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        data = request.json.get("features")
        if not data:
            print("No input data provided")
            return jsonify({"error": "No input data provided"}), 400

        input_df = pd.DataFrame([data], columns=preprocessor.feature_names_in_)
        input_transformed = preprocessor.transform(input_df)
        prediction = model.predict(input_transformed)

        label_mapping = {0: "N", 1: "Y"} 
        prediction_labels = [label_mapping[pred] for pred in prediction]
        return jsonify({"prediction": prediction_labels})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
