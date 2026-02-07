
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}

    def handle_imbalance(self, X_train, y_train):
        print("Handling class imbalance with SMOTE...")
        smote = SMOTE(random_state=self.random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print("✓ Applied SMOTE")
        print(f"  Before: {y_train.value_counts().to_dict()}")
        print(f"  After: {y_train_res.value_counts().to_dict()}")
        return X_train_res, y_train_res

    def train_logistic_regression(self, X_train, y_train):
        print("Training Logistic Regression...")
        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr
        print("✓ Trained Logistic Regression")
        return lr

    def train_random_forest(self, X_train, y_train):
        print("Training Random Forest...")
        rf = RandomForestClassifier(random_state=self.random_state)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        print("✓ Trained Random Forest")
        return rf

    def train_xgboost(self, X_train, y_train):
        print("Training XGBoost...")
        xgb = XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        self.models['XGBoost'] = xgb
        print("✓ Trained XGBoost")
        return xgb

    def train_lightgbm(self, X_train, y_train):
        print("Training LightGBM...")
        lgbm = LGBMClassifier(random_state=self.random_state)
        lgbm.fit(X_train, y_train)
        self.models['LightGBM'] = lgbm
        print("✓ Trained LightGBM")
        return lgbm

    def tune_hyperparameters(self, model, X_train, y_train, param_grid):
        print(f"Tuning hyperparameters for {model.__class__.__name__}...")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        print(f"✓ Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

if __name__ == '__main__':
    from data_loader import LoanDataGenerator
    from feature_engineering import FeatureEngineer

    # Generate and prepare data
    generator = LoanDataGenerator(n_samples=1000)
    data = generator.generate_data()
    X_train_raw, _, _, y_train, _, _ = generator.split_data()
    
    feature_engineer = FeatureEngineer()
    X_train = feature_engineer.fit_transform(X_train_raw)

    # Train models
    trainer = ModelTrainer()
    X_train_res, y_train_res = trainer.handle_imbalance(X_train, y_train)
    
    trainer.train_logistic_regression(X_train_res, y_train_res)
    trainer.train_random_forest(X_train_res, y_train_res)
    trainer.train_xgboost(X_train_res, y_train_res)
    trainer.train_lightgbm(X_train_res, y_train_res)

    print("\nModel training complete.")
    print("Models trained:", list(trainer.models.keys()))
