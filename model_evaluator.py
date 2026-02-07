import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class ModelEvaluator:
    def __init__(self, models=None):
        self.models = models if models is not None else {}

    def evaluate_models(self, X_test, y_test, save_path='./visualizations/'):
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'AUC-ROC': roc_auc_score(y_test, y_proba)
            }
        
        results_df = pd.DataFrame(results).T
        print("\nModel Performance Comparison:")
        print(results_df)

        self.plot_roc_curves(X_test, y_test, save_path)
        
        # Determine best model based on AUC-ROC
        best_model_name = results_df['AUC-ROC'].idxmax()
        best_model = self.models[best_model_name]
        print(f"\nBest Model: {best_model_name}")

        self.plot_confusion_matrix(best_model, X_test, y_test, save_path)
        self.plot_feature_importance(best_model, X_test.columns, save_path)

        # Only plot SHAP for tree-based models
        if isinstance(best_model, (RandomForestClassifier, XGBClassifier, LGBMClassifier)):
            self.plot_shap_summary(best_model, X_test, save_path)
        else:
            print(f"Skipping SHAP summary plot for {best_model_name} as it's not a tree-based model compatible with TreeExplainer.")

        return best_model_name, best_model

    def plot_roc_curves(self, X_test, y_test, save_path):
        plt.figure(figsize=(10, 8))
        for name, model in self.models.items():
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig(f'{save_path}roc_curves.png')
        plt.close()

    def plot_confusion_matrix(self, model, X_test, y_test, save_path):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model.__class__.__name__}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{save_path}confusion_matrix.png')
        plt.close()

    def plot_feature_importance(self, model, feature_names, save_path):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-20:]
            plt.figure(figsize=(10, 12))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.savefig(f'{save_path}feature_importance.png')
            plt.close()
        else:
            print(f"Skipping feature importance plot for {model.__class__.__name__} as it does not have 'feature_importances_'.")


    def plot_shap_summary(self, model, X_test, save_path):
        # TreeExplainer is suitable for tree-based models.
        # For other models, KernelExplainer or LinearExplainer might be needed.
        # Given the context, we're expecting tree-based models to be best.
        if isinstance(model, (RandomForestClassifier, XGBClassifier, LGBMClassifier)):
            explainer = shap.TreeExplainer(model)
            # Take a sample of X_test for SHAP to speed up computation if X_test is very large
            if X_test.shape[0] > 1000:
                X_sample = X_test.sample(n=1000, random_state=42)
            else:
                X_sample = X_test
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.savefig(f'{save_path}shap_summary.png', bbox_inches='tight')
            plt.close()
        else:
            print(f"Skipping SHAP summary plot for {model.__class__.__name__} as it's not a tree-based model compatible with TreeExplainer.")


if __name__ == '__main__':
    from data_loader import LoanDataGenerator
    from feature_engineering import FeatureEngineer
    from model_trainer import ModelTrainer

    # Generate and prepare data
    generator = LoanDataGenerator(n_samples=1000, random_state=42)
    data = generator.generate_data()
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = generator.split_data()
    
    feature_engineer = FeatureEngineer()
    X_train = feature_engineer.fit_transform(X_train_raw)
    X_test = feature_engineer.transform(X_test_raw)

    # Align columns after dummy variable creation
    train_cols = X_train.columns
    test_cols = X_test.columns

    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
    X_test = X_test[train_cols]


    # Train models
    trainer = ModelTrainer()
    X_train_res, y_train_res = trainer.handle_imbalance(X_train, y_train)
    
    models = {
        'Logistic Regression': trainer.train_logistic_regression(X_train_res, y_train_res),
        'Random Forest': trainer.train_random_forest(X_train_res, y_train_res),
        'XGBoost': trainer.train_xgboost(X_train_res, y_train_res),
        'LightGBM': trainer.train_lightgbm(X_train_res, y_train_res)
    }

    # Evaluate models
    evaluator = ModelEvaluator(models)
    best_model_name, best_model = evaluator.evaluate_models(X_test, y_test)
    print("\nModel evaluation complete. Plots saved to ./visualizations/")
    print(f"The best model is {best_model_name}")