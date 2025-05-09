import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, average_precision_score, 
    roc_curve, roc_auc_score 
)
from preprocessing_pipeline import PreprocessingPipeline

class DepositSubscriptionClassifier:
    """
    A classifier class to train, evaluate, save, load, and predict term deposit subscriptions.
    Includes preprocessing, model training, evaluation with adjustable threshold,
    and visualization (Confusion Matrix, Feature Importance, Precision-Recall Curve).
    """
    def __init__(self,
                 transform_features=None, 
                 numeric_features=None,  
                 categorical_features=None,
                 target=None,
                 scaler_type='standard', 
                 models_dir='marketing_saved_models'):

        if target is None:
            raise ValueError("Target column name must be provided.")

        self.target = target
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

        self.preprocessor = PreprocessingPipeline(
            transform_features=transform_features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scaler_type=scaler_type
        )

        # Feature lists used only for selecting columns from input data
        self.all_input_features = (transform_features or []) + (numeric_features or []) + (categorical_features or [])
        self.feature_names_after_transform = None # Populated after fitting preprocessor

        self.models = {}
        self.best_model = None 
        # Using F1 Macro @ 0.5 threshold for initial best model selection during train_models
        self.best_model_info = {'name': None, 'score': -np.inf, 'metric': 'f1_macro_at_0.5'}

    def _prepare_data(self, data):
        """Selects feature and target columns from the input DataFrame."""
        missing_cols = [col for col in self.all_input_features if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in input data needed for preprocessing: {missing_cols}")
        if self.target not in data.columns:
             raise ValueError(f"Target column '{self.target}' not found in data.")

        # Returns only the necessary features for the preprocessor
        return data[self.all_input_features], data[self.target]

    def evaluate_model(self, model, X, y, label="Test", threshold=0.5):
        """
        Evaluate classification model performance using a specific threshold.
        Calculates standard metrics, Precision-Recall curve data, Average Precision,
        ROC curve data, and AUC score.
        """
        print(f"\n--- {label} Set Performance (Threshold: {threshold}) ---")
    
        metrics = {
            'threshold': threshold,
            'accuracy': None,
            'f1_macro': None,
            'classification_report': None,
            'confusion_matrix': None,
            'precision_points': None,
            'recall_points': None,
            'average_precision': None,
            'fpr': None, # Added for ROC
            'tpr': None, # Added for ROC
            'roc_auc_score': None # Added for ROC
        }
        model_classes = getattr(model, 'classes_', [0, 1])
    
        if hasattr(model, 'predict_proba'):
            try:
                positive_class_index = np.where(model_classes == 1)[0]
                if len(positive_class_index) == 0: raise ValueError("Positive class (1) not found.")
                positive_class_index = positive_class_index[0]
                y_pred_proba = model.predict_proba(X)[:, positive_class_index]
    
                # Apply threshold for classification metrics
                y_pred = (y_pred_proba >= threshold).astype(int)
    
                metrics['accuracy'] = accuracy_score(y, y_pred)
                metrics['f1_macro'] = f1_score(y, y_pred, average='macro', zero_division=0)
                metrics['classification_report'] = classification_report(y, y_pred, zero_division=0, target_names=[f'Class {c}' for c in model_classes])
                metrics['confusion_matrix'] = confusion_matrix(y, y_pred, labels=model_classes)
    
                # Calculate PR Curve data
                metrics['precision_points'], metrics['recall_points'], _ = precision_recall_curve(y, y_pred_proba)
                metrics['average_precision'] = average_precision_score(y, y_pred_proba)
    
                # Calculate ROC Curve data
                metrics['fpr'], metrics['tpr'], _ = roc_curve(y, y_pred_proba)
                metrics['roc_auc_score'] = roc_auc_score(y, y_pred_proba)
    
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
                print(f"Average Precision (AP): {metrics['average_precision']:.4f}")
                print(f"AUC Score: {metrics['roc_auc_score']:.4f}") # Print AUC
                print("Classification Report:\n", metrics['classification_report'])
    
            except Exception as e:
                print(f"Could not calculate metrics with probabilities: {e}")
                # Fallback calculation (remains the same)
                y_pred = model.predict(X)
                metrics['accuracy'] = accuracy_score(y, y_pred)
                metrics['f1_macro'] = f1_score(y, y_pred, average='macro', zero_division=0)
                metrics['classification_report'] = classification_report(y, y_pred, zero_division=0)
                metrics['confusion_matrix'] = confusion_matrix(y, y_pred)
                print("Using default predict() for metrics.")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
                print("Classification Report:\n", metrics['classification_report'])
    
        else: # Model does not support predict_proba (remains the same)
             print("Model does not support predict_proba. Using default threshold (0.5 implicitly).")
             y_pred = model.predict(X)
             metrics['accuracy'] = accuracy_score(y, y_pred)
             metrics['f1_macro'] = f1_score(y, y_pred, average='macro', zero_division=0)
             metrics['classification_report'] = classification_report(y, y_pred, zero_division=0)
             metrics['confusion_matrix'] = confusion_matrix(y, y_pred)
             print(f"Accuracy: {metrics['accuracy']:.4f}")
             print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
             print("Classification Report:\n", metrics['classification_report'])
    
        return metrics


    def train_models(self, train_data, test_size=0.2, random_state=42, stratify=True):
        """
        Trains and evaluates multiple classification models. Selects best model based
        on F1 Macro score at the default 0.5 threshold during this run.
        Allows manual evaluation with different thresholds afterwards.
        """
        X, y = self._prepare_data(train_data)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if stratify else None
        )

        print(f"Original Training Features shape: {X_train.shape}")
        print(f"Original Testing Features shape: {X_test.shape}")

        print("\nFitting and transforming data using PreprocessingPipeline...")
        # Fits preprocessor on training data and transform both sets
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # Gets feature names AFTER fitting the preprocessor
        try:
            self.feature_names_after_transform = self.preprocessor.transformer.get_feature_names_out()
            print("Successfully retrieved feature names after transformation.")
        except Exception as e:
            print(f"Warning: Could not retrieve feature names automatically: {e}")
            self.feature_names_after_transform = None

        print("Preprocessing complete.")
        print(f"Processed Training Features shape: {X_train_processed.shape}")
        print(f"Processed Testing Features shape: {X_test_processed.shape}")

        # --- Model Definitions ---
        model_definitions = {
            'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=5, random_state=random_state),
            'XGBoost': XGBClassifier(
                random_state=random_state,
                eval_metric='logloss',
                learning_rate=0.05,
                max_depth=6,
                n_estimators=100,
                subsample=1,
                colsample_bytree=1
            ),
        }

        # --- Training Loop ---
        for name, model in model_definitions.items():
            print(f"\n{'='*10} Training: {name} {'='*10}")

            try:
                model.fit(X_train_processed, y_train)
                self.models[name] = model 
                print("\n--- Evaluating with Default Threshold (0.5) ---")
                train_metrics_default = self.evaluate_model(model, X_train_processed, y_train, "Training", threshold=0.5)
                test_metrics_default = self.evaluate_model(model, X_test_processed, y_test, "Test", threshold=0.5)

                # --- Visualizes results based on DEFAULT threshold evaluation ---
                self.visualize_results(model, test_metrics_default, name)

                # --- Selects best model based on F1 Macro at DEFAULT threshold ---
                current_f1_macro_default = test_metrics_default.get('f1_macro', -1)
                if current_f1_macro_default > self.best_model_info['score']:
                     self.best_model_info = {'name': name, 'score': current_f1_macro_default, 'metric': 'f1_macro_at_0.5'}
                     self.best_model = (name, model)
                     print(f"*** New best model (based on F1 Macro @ 0.5 thresh): {name} with F1: {current_f1_macro_default:.4f} ***")

            except Exception as e:
                 print(f"!!! Failed to train or evaluate {name}: {e} !!!")


        print(f"\nFinal Best model (based on {self.best_model_info['metric']}): {self.best_model_info['name']} ({self.best_model_info['score']:.4f})")
        self.save_models() 

        self.X_test_processed_ = X_test_processed
        self.y_test_ = y_test
        return self.models

    def visualize_results(self, model, test_metrics, model_name):
        """Visualizes CM, Feature Importance, PR Curve, and ROC Curve in a 2x2 grid."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        eval_threshold = test_metrics.get('threshold', 0.5)
        fig.suptitle(f'Evaluation Metrics for {model_name} (Threshold: {eval_threshold})', fontsize=16)
        model_classes = getattr(model, 'classes_', [0, 1])
    
        # --- Plot 1: Confusion Matrix (Top-Left: axes[0, 0]) ---
        ax_cm = axes[0, 0]
        try:
            cm_display = test_metrics.get('confusion_matrix')
            if cm_display is not None:
                sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                            xticklabels=model_classes, yticklabels=model_classes)
            else:
                ax_cm.text(0.5, 0.5, 'Confusion Matrix N/A', ha='center', va='center')
            ax_cm.set_title('Confusion Matrix (Test Set)')
            ax_cm.set_ylabel('Actual')
            ax_cm.set_xlabel('Predicted')
        except Exception as e:
            ax_cm.text(0.5, 0.5, f'Could not plot CM:\n{e}', ha='center', va='center')
            ax_cm.set_title('Confusion Matrix (Plot Error)')
    
        # --- Plot 2: Feature Importance (Top-Right: axes[0, 1]) ---
        ax_fi = axes[0, 1]
        importances = None
        feature_names = self.feature_names_after_transform
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_.flatten())
    
        if importances is not None and feature_names is not None:
            if len(importances) == len(feature_names):
                feat_imp = pd.DataFrame({'Feature': feature_names, 'Value': importances})
                feat_imp = feat_imp.dropna().sort_values('Value', ascending=False).head(10)
                if not feat_imp.empty and feat_imp['Value'].sum() > 0:
                     sns.barplot(x='Value', y='Feature', data=feat_imp.sort_values('Value', ascending=False),
                                 ax=ax_fi, color='skyblue')
                     ax_fi.tick_params(axis='y', labelsize=9)
                else:
                     ax_fi.text(0.5, 0.5, 'No non-zero feature importances', ha='center', va='center')
                ax_fi.set_title('Top 10 Feature Importances')
                ax_fi.set_ylabel('')
                ax_fi.set_xlabel('')
            else:
                ax_fi.text(0.5, 0.5, f'Importance/Name mismatch ({len(importances)} vs {len(feature_names)})',
                                ha='center', va='center', wrap=True)
                ax_fi.set_title('Feature Importance Error')
        else:
            ax_fi.text(0.5, 0.5, 'Feature importances not available',
                        ha='center', va='center', fontsize=12, color='grey')
            ax_fi.set_title('Feature Importances')
    
        # --- Plot 3: Precision-Recall Curve (Bottom-Left: axes[1, 0]) ---
        ax_pr = axes[1, 0]
        if test_metrics.get('precision_points') is not None and test_metrics.get('recall_points') is not None:
            precision = test_metrics['precision_points']
            recall = test_metrics['recall_points']
            ap_score = test_metrics.get('average_precision', None)
    
            if len(precision) == len(recall) + 1:
                precision = precision[:-1] 
    
            ax_pr.plot(recall, precision, color='purple', lw=2,
                         label=f'PR curve (AP = {ap_score:.2f})' if ap_score is not None else 'PR curve')
            ax_pr.set_xlabel('Recall (Positive Class: 1)')
            ax_pr.set_ylabel('Precision (Positive Class: 1)')
            ax_pr.set_title('Precision-Recall Curve')
            ax_pr.set_ylim([0.0, 1.05])
            ax_pr.set_xlim([0.0, 1.0])
            ax_pr.legend(loc="lower left")
            ax_pr.grid(alpha=0.3)
        else:
            ax_pr.text(0.5, 0.5, 'PR Curve not available\n(requires predict_proba)',
                        ha='center', va='center', fontsize=12, color='grey')
            ax_pr.set_title('Precision-Recall Curve')
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            ax_pr.set_ylim([0.0, 1.05])
            ax_pr.set_xlim([0.0, 1.0])
    
        # --- Plot 4: ROC Curve (Bottom-Right: axes[1, 1]) ---
        ax_roc = axes[1, 1]
        if test_metrics.get('fpr') is not None and test_metrics.get('tpr') is not None:
            fpr = test_metrics['fpr']
            tpr = test_metrics['tpr']
            auc_score = test_metrics.get('roc_auc_score', None)
    
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2,
                         label=f'ROC curve (AUC = {auc_score:.2f})' if auc_score is not None else 'ROC curve')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal reference line
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic (ROC)')
            ax_roc.legend(loc="lower right")
            ax_roc.grid(alpha=0.3)
        else:
            ax_roc.text(0.5, 0.5, 'ROC Curve not available\n(requires predict_proba)',
                        ha='center', va='center', fontsize=12, color='grey')
            ax_roc.set_title('ROC Curve')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlim([0.0, 1.0])
    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.show()

    def save_models(self):
        """Save the preprocessor and all trained models."""
        if not self.models or not self.preprocessor:
            print("No models or preprocessor to save.")
            return

        self.preprocessor.save(os.path.join(self.models_dir, 'preprocessor_pipeline.joblib'))
        joblib.dump(self.best_model_info, os.path.join(self.models_dir, 'best_model_info.joblib'))
        # Saves processed feature names if available
        if self.feature_names_after_transform is not None:
             joblib.dump(self.feature_names_after_transform, os.path.join(self.models_dir, 'feature_names.joblib'))


        for name, model in self.models.items():
            joblib.dump(model, os.path.join(self.models_dir, f'{name}_model.joblib'))

        print(f"\nSaved PreprocessingPipeline, best model info, feature names and {len(self.models)} models to {self.models_dir}")

    def load_models(self):
        """Load the preprocessor and all trained models."""
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor_pipeline.joblib')
        best_model_info_path = os.path.join(self.models_dir, 'best_model_info.joblib')
        feature_names_path = os.path.join(self.models_dir, 'feature_names.joblib')


        required_files = [preprocessor_path, best_model_info_path] # Feature names optional but good to have
        if not all(os.path.exists(p) for p in required_files):
             raise FileNotFoundError("Required files (preprocessor_pipeline.joblib, best_model_info.joblib) not found.")

        self.preprocessor = PreprocessingPipeline.load(preprocessor_path)
        self.best_model_info = joblib.load(best_model_info_path)
        try:
            self.feature_names_after_transform = joblib.load(feature_names_path)
        except FileNotFoundError:
            print("Warning: feature_names.joblib not found. Feature names will be unavailable after loading.")
            self.feature_names_after_transform = None


        self.models = {}
        for file_name in os.listdir(self.models_dir):
            if file_name.endswith('_model.joblib'):
                name = file_name.replace('_model.joblib', '')
                try:
                    self.models[name] = joblib.load(os.path.join(self.models_dir, file_name))
                except Exception as e:
                    print(f"Warning: Could not load model file {file_name}: {e}")

        if self.best_model_info['name'] in self.models:
            self.best_model = (self.best_model_info['name'], self.models[self.best_model_info['name']])
        else:
             print(f"Warning: Best model '{self.best_model_info['name']}' not found among loaded models.")
             self.best_model = None

        print(f"Loaded PreprocessingPipeline, best model info, feature names and {len(self.models)} models from {self.models_dir}")
        return self.models

    def predict(self, new_data, model_name=None, threshold=0.5): # Default threshold to predict
        """Makes predictions on new data using a specified or the best model and a specified threshold."""
        if not hasattr(self.preprocessor, 'transformer') or not self.models:
            print("Models or preprocessor not available. Loading models...")
            self.load_models()

        # Determines model to use
        model_to_use = None
        chosen_model_name = None
        if model_name:
            if model_name in self.models:
                model_to_use = self.models[model_name]
                chosen_model_name = model_name
            else:
                raise ValueError(f"Model '{model_name}' not found.")
        elif self.best_model:
            model_to_use = self.best_model[1]
            chosen_model_name = self.best_model[0]
        else:
            if self.models:
                 chosen_model_name = next(iter(self.models))
                 model_to_use = self.models[chosen_model_name]
                 print(f"Warning: Best model not set. Using first loaded model: {chosen_model_name}")
            else:
                 raise RuntimeError("No models available for prediction.")

        print(f"Making predictions using model: {chosen_model_name} with threshold: {threshold}")

        temp_data = new_data.copy()
        if self.target not in temp_data.columns:
             temp_data[self.target] = 0 
        X_features_only, _ = self._prepare_data(temp_data)
        X_processed = self.preprocessor.transform(X_features_only)


        if hasattr(model_to_use, 'predict_proba'):
            positive_class_index = np.where(model_to_use.classes_ == 1)[0][0]
            probabilities = model_to_use.predict_proba(X_processed)
            predictions = (probabilities[:, positive_class_index] >= threshold).astype(int)
            proba_col_0 = f'Probability_{model_to_use.classes_[0]}'
            proba_col_1 = f'Probability_{model_to_use.classes_[1]}'
        else:
            print(f"Warning: Model {chosen_model_name} does not support predict_proba. Using predict() (threshold ignored).")
            predictions = model_to_use.predict(X_processed)
            probabilities = None 

        result_df = new_data.copy() 
        result_df[f'Predicted_{self.target}'] = predictions
        if probabilities is not None:
            result_df[proba_col_0] = probabilities[:, 0] # Probability of class 0
            result_df[proba_col_1] = probabilities[:, 1] # Probability of class 1

        return result_df