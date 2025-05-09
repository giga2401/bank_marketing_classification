import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
import joblib
import os

def log_transform_func(x):
    return np.log1p(x)

class PreprocessingPipeline:
    """Handles preprocessing including log transforms, imputation, scaling, and encoding."""

    def __init__(self, transform_features=None, numeric_features=None, categorical_features=None, scaler_type='standard'):
        
        self.log_features = transform_features or []
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.scaler_type = scaler_type
        self._validate_features()

        self.transformer = self._build_transformer()
        self.feature_names_out_ = None 

    def _validate_features(self):
        """Ensure feature lists are disjoint."""
        all_numeric = set(self.log_features) | set(self.numeric_features)
        if len(all_numeric) != (len(self.log_features) + len(self.numeric_features)):
            overlapping = set(self.log_features) & set(self.numeric_features)
            raise ValueError(f"Features cannot be in both 'log_features' and 'numeric_features'. Overlapping: {overlapping}")
        if set(all_numeric) & set(self.categorical_features):
             overlapping_cat = set(all_numeric) & set(self.categorical_features)
             raise ValueError(f"Features cannot be both numeric (log or standard) and categorical. Overlapping: {overlapping_cat}")

    def _build_transformer(self):
        """Constructs the ColumnTransformer."""
        transformers = []
        scaler = StandardScaler() if self.scaler_type == 'standard' else MinMaxScaler()
    
        # Pipeline for features needing log transform first
        if self.log_features:
            log_pipeline = Pipeline(steps=[
                ('imputer_log', SimpleImputer(strategy='median')),
                ('log_transform', FunctionTransformer(
                    log_transform_func, 
                    validate=False,
                    feature_names_out="one-to-one")),  #preserves feature names
                ('scaler_log', scaler)
            ])
            transformers.append(('log_num', log_pipeline, self.log_features))
    
        # Pipeline for other numeric features
        if self.numeric_features:
            numeric_pipeline = Pipeline(steps=[
                ('imputer_num', SimpleImputer(strategy='median')),
                ('scaler_num', scaler)
            ])
            transformers.append(('num', numeric_pipeline, self.numeric_features))
    
        # Pipeline for categorical features
        if self.categorical_features:
            categorical_pipeline = Pipeline(steps=[
                ('imputer_cat', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
            ])
            transformers.append(('cat', categorical_pipeline, self.categorical_features))
    
        if not transformers:
             raise ValueError("No features specified for preprocessing.")
    
        preprocessor = ColumnTransformer(
            transformers=transformers, 
            remainder='drop', 
            verbose_feature_names_out=False
        )
        preprocessor.set_output(transform="pandas")
        return preprocessor

    def fit(self, X):
        """Fits the transformer to the data X."""
        self.transformer.fit(X)
        self.feature_names_out_ = self.transformer.get_feature_names_out()
        return self

    def transform(self, X):
        """Transforms the data X using the fitted transformer."""
        if self.feature_names_out_ is None:
            raise RuntimeError("Transformer must be fitted before transforming.")
        # Ensures output is pandas DataFrame with correct columns
        return self.transformer.transform(X)

    def fit_transform(self, X):
        """Fits and transforms the data X."""
        transformed_data = self.transformer.fit_transform(X)
        # Stores feature names after fitting
        self.feature_names_out_ = self.transformer.get_feature_names_out()
        return transformed_data

    def get_feature_names_out(self):
        """Returns the output feature names after transformation."""
        if self.feature_names_out_ is None:
             raise RuntimeError("Transformer must be fitted before getting feature names.")
        return self.feature_names_out_

    def save(self, filepath):
        """Saves the fitted transformer."""
        joblib.dump(self, filepath)
        print(f"PreprocessingPipeline saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Loads a saved transformer."""
        if not os.path.exists(filepath):
             raise FileNotFoundError(f"File not found: {filepath}")
        loaded_pipeline = joblib.load(filepath)
        print(f"PreprocessingPipeline loaded from {filepath}")
        return loaded_pipeline