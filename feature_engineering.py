
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    def __init__(self, numerical_cols=None, categorical_cols=None):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler = StandardScaler()
        self.imputer_numerical = SimpleImputer(strategy='median')
        self.imputer_categorical = SimpleImputer(strategy='most_frequent')

    def fit_transform(self, df):
        if self.numerical_cols is None:
            self.numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if self.categorical_cols is None:
            self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        df = self.create_ratio_features(df)
        df = self.create_interaction_features(df)

        # Impute missing values
        df[self.numerical_cols] = self.imputer_numerical.fit_transform(df[self.numerical_cols])
        if self.categorical_cols:
            df[self.categorical_cols] = self.imputer_categorical.fit_transform(df[self.categorical_cols])

        # Scale numerical features
        df[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])

        # Encode categorical features
        if self.categorical_cols:
            df = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)

        return df

    def transform(self, df):
        df = self.create_ratio_features(df)
        df = self.create_interaction_features(df)

        # Impute missing values
        df[self.numerical_cols] = self.imputer_numerical.transform(df[self.numerical_cols])
        if self.categorical_cols:
            df[self.categorical_cols] = self.imputer_categorical.transform(df[self.categorical_cols])
        
        # Scale numerical features
        df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])

        # Encode categorical features
        if self.categorical_cols:
            df = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)

        return df

    def create_ratio_features(self, df):
        df['debt_to_income'] = df['existing_debt'] / (df['income'] + 1)
        df['loan_to_income'] = df['loan_amount'] / (df['income'] + 1)
        df['payment_to_income'] = df['monthly_payment'] / (df['income'] / 12 + 1)
        df['credit_utilization'] = df['credit_used'] / (df['credit_limit'] + 1)
        return df

    def create_interaction_features(self, df):
        df['age_income'] = df['age'] * df['income']
        df['credit_score_loan_amount'] = df['credit_score'] * df['loan_amount']
        return df

if __name__ == '__main__':
    from data_loader import LoanDataGenerator

    # Generate data
    generator = LoanDataGenerator(n_samples=1000)
    data = generator.generate_data()
    X = data.drop('default', axis=1)

    # Engineer features
    feature_engineer = FeatureEngineer()
    X_engineered = feature_engineer.fit_transform(X)

    print("Feature engineering complete.")
    print("Original number of features:", X.shape[1])
    print("Number of features after engineering:", X_engineered.shape[1])
    print("\nEngineered features preview:")
    print(X_engineered.head())
