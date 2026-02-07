
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
        best_model_name = results_df['AUC-ROC'].idxmax()
        best_model = self.models[best_model_name]

        self.plot_confusion_matrix(best_model, X_test, y_test, save_path)
        self.plot_feature_importance(best_model, X_test.columns, save_path)
        self.plot_shap_summary(best_model, X_test, save_path)

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

    def plot_shap_summary(self, model, X_test, save_path):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(f'{save_path}shap_summary.png', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    from data_loader import LoanDataGenerator
    from feature_engineering import FeatureEngineer
    from model_trainer import ModelTrainer

    # Generate and prepare data
    generator = LoanDataGenerator(n_samples=1000)
    data = generator.generate_data()
    X_train_raw, X_test_raw, _, y_train, y_test, _ = generator.split_data()
    
    feature_engineer = FeatureEngineer()
    X_train = feature_engineer.fit_transform(X_train_raw)
    X_test = feature_engineer.transform(X_test_raw)

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
    evaluator.evaluate_models(X_test, y_test)
    print("\nModel evaluation complete. Plots saved to ./visualizations/")
