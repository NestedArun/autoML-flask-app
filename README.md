# AutoML Flask Application

This web application uses the concepts of Automated Machine Learning (AutoML). Users can upload datasets and train machine learning models for both **regression** and **classification** tasks. The app automatically selects appropriate algorithms, trains the model, and evaluates its performance.

## Models Used

### Regression Models:
- **RandomForest**
- **GradientBoosting**
- **LinearRegression**
- **SVR**

### Classification Models:
- **RandomForest**
- **GradientBoosting**
- **LogisticRegression**
- **SVC**
  
## Features
- Upload datasets in CSV format
- Automatically chooses the best model and trains it
- Evaluates model performance (R² score for regression and Accuracy score for classification)
- The best model and its score are shown  
- Save trained models for future use

## Requirements
- Python 3.x
- Flask
- scikit-learn
- joblib

## Installation
1. Clone this repository:
   ```cmd
   git clone https://github.com/NestedArun/autoML-flask-app.git
   ```
2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
3. Run the app:
   ```cmd
   python run.py
   ```

## Usage
- Visit the app in your browser (default: `http://127.0.0.1:5000`).
- Upload your dataset.
- Enter the input data
- Get output

## License
MIT License
