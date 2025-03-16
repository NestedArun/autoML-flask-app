import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformation:
    def __init__(self, target_column):
        self.target_column = target_column

    def preprocess_data(self, df):
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        df = df.dropna(thresh=len(df) * 0.6, axis=1)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        non_numerical_columns = X.select_dtypes(include=['object']).columns.tolist()
        text_columns = [col for col in non_numerical_columns if X[col].nunique() > 50]  
        X = X.drop(columns=text_columns)  

        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features)
        ])

        X_transformed = preprocessor.fit_transform(X)

        if y.dtype == 'object' or y.nunique() <= 10:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            joblib.dump(label_encoder, "api/models/label_encoder.pkl")  # Save label encoder

        joblib.dump(preprocessor, "api/models/preprocessor.pkl")

        return X_transformed, y, preprocessor
