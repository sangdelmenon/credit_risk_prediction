
import pickle
from data_loader import LoanDataGenerator
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

class CreditRiskPipeline:
    def __init__(self, n_samples=100000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        self.data_generator = LoanDataGenerator(n_samples=self.n_samples, random_state=self.random_state)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(random_state=self.random_state)
        self.model_evaluator = None
        self.best_model = None

    def run_complete_pipeline(self):
        # 1. Load and split data
        print("--- Step 1: Loading Data ---")
        data = self.data_generator.generate_data()
        self.data_generator.perform_eda()
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = self.data_generator.split_data()

        # 2. Feature Engineering
        print("\n--- Step 2: Feature Engineering ---")
        X_train = self.feature_engineer.fit_transform(X_train_raw.copy())
        X_val = self.feature_engineer.transform(X_val_raw.copy())
        X_test = self.feature_engineer.transform(X_test_raw.copy())

        # Align columns after dummy variable creation
        train_cols = X_train.columns
        val_cols = X_val.columns
        test_cols = X_test.columns

        missing_in_val = set(train_cols) - set(val_cols)
        for c in missing_in_val:
            X_val[c] = 0
        X_val = X_val[train_cols]

        missing_in_test = set(train_cols) - set(test_cols)
        for c in missing_in_test:
            X_test[c] = 0
        X_test = X_test[train_cols]

        # 3. Handle Class Imbalance
        print("\n--- Step 3: Handling Class Imbalance ---")
        X_train_res, y_train_res = self.model_trainer.handle_imbalance(X_train, y_train)

        # 4. Model Training
        print("\n--- Step 4: Model Training ---")
        models = {
            'Logistic Regression': self.model_trainer.train_logistic_regression(X_train_res, y_train_res),
            'Random Forest': self.model_trainer.train_random_forest(X_train_res, y_train_res),
            'XGBoost': self.model_trainer.train_xgboost(X_train_res, y_train_res),
            'LightGBM': self.model_trainer.train_lightgbm(X_train_res, y_train_res)
        }
        self.model_evaluator = ModelEvaluator(models)

        # 5. Model Evaluation
        print("\n--- Step 5: Model Evaluation ---")
        best_model_name, self.best_model = self.model_evaluator.evaluate_models(X_test, y_test)
        print(f"\nBest Model: {best_model_name}")

        # 6. Save Best Model
        print("\n--- Step 6: Saving Best Model ---")
        self.save_model(self.best_model, './models/best_model.pkl')

    def save_model(self, model, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Model saved to {file_path}")

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

if __name__ == '__main__':
    pipeline = CreditRiskPipeline()
    pipeline.run_complete_pipeline()
    print("\nCredit Risk Prediction pipeline finished successfully!")
