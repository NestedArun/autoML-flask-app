import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformation:
    def __init__(self, target_column):
        self.target_column = target_column

    def preprocess_data(self, df):
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

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

        joblib.dump(preprocessor, "api/models/preprocessor.pkl")

        return X_transformed, y, preprocessor
