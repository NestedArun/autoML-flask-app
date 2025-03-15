from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import joblib
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

        # Data Transformation
        data_transformation = DataTransformation(target_column)
        X, y, preprocessor = data_transformation.preprocess_data(df)
        print("Data Preprocessed Successfully")

        # Model Training
        model_trainer = ModelTrainer()
        performance, model = model_trainer.train_model(X, y)
        print(f"Model Trained Successfully with RMSE: {performance}")

        # Save model and preprocessor
        model_path = os.path.join(MODEL_FOLDER, "model.pkl")
        preprocessor_path = os.path.join(MODEL_FOLDER, "preprocessor.pkl")

        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        print(f"Model saved successfully at: {model_path}")
        return render_template("index.html", trained=True)

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    try:
        model_path = os.path.join(MODEL_FOLDER, "model.pkl")
        preprocessor_path = os.path.join(MODEL_FOLDER, "preprocessor.pkl")

        if not os.path.exists(model_path):
            print("Model file missing!")
            return jsonify({"error": "No trained model found"}), 400

        if not os.path.exists(preprocessor_path):
            print(" Preprocessor file missing!")
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

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
