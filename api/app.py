from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import joblib
import json
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from sklearn.preprocessing import LabelEncoder

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
        task_type = "classification" if y_series.nunique() <= 10 and y_series.dtype in ['int64', 'int32', 'object'] else "regression"

        label_mapping = None
        if task_type == "classification":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_series)  # Convert categorical labels to numbers
            label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}  # Store label mapping

        model_trainer = ModelTrainer()
        best_model_name, best_score, best_model = model_trainer.train_best_model(X, y, task_type)
        print(f"Best Model Selected: {best_model_name} with Score: {best_score}")

        model_path = os.path.join(MODEL_FOLDER, "best_model.pkl")
        preprocessor_path = os.path.join(MODEL_FOLDER, "preprocessor.pkl")

        joblib.dump(best_model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        print(f"Best Model saved successfully at: {model_path}")

        # Store model metadata, including label mappings for classification models
        best_model_data = {
            "best_model": best_model_name,
            "score": float(best_score),
            "task_type": task_type,
            "target_type": "int" if y_series.dtype == "int64" else "float",
            "label_mapping": label_mapping  # Store label mapping for classification
        }
        with open(os.path.join(MODEL_FOLDER, "best_model.json"), "w") as f:
            json.dump(best_model_data, f)

        return jsonify(best_model_data)

    except Exception as e:
        print(f"Error during training: {str(e)}")  
        return jsonify({"error": str(e)}), 500  

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    try:
        model_path = os.path.join(MODEL_FOLDER, "best_model.pkl")
        preprocessor_path = os.path.join(MODEL_FOLDER, "preprocessor.pkl")
        model_info_path = os.path.join(MODEL_FOLDER, "best_model.json")

        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path) or not os.path.exists(model_info_path):
            print("Required files missing!")
            return jsonify({"error": "No trained model or preprocessor found"}), 400

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        with open(model_info_path, "r") as f:
            model_info = json.load(f)
            task_type = model_info.get("task_type", "regression") 
            target_type = model_info.get("target_type", "float") 
            label_mapping = model_info.get("label_mapping", None)  

        data = request.json.get("features")
        if not data:
            print("No input data provided")
            return jsonify({"error": "No input data provided"}), 400

        input_df = pd.DataFrame([data], columns=preprocessor.feature_names_in_)
        input_transformed = preprocessor.transform(input_df)
        prediction = model.predict(input_transformed)

        if task_type == "classification":
            if label_mapping:
                prediction_labels = [label_mapping.get(int(pred), str(pred)) for pred in prediction]
            else:
                prediction_labels = prediction.tolist()  
            return jsonify({"prediction": prediction_labels})

        else:  
            formatted_prediction = int(prediction[0]) if target_type == "int" else float(prediction[0])
            return jsonify({"prediction": formatted_prediction})  

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == "__main__":
    app.run(debug=True)
