
"""
Credit Risk Prediction - Data Loader Module
Generates synthetic loan data and performs EDA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Tuple


class LoanDataGenerator:
    """Generate realistic synthetic loan application data"""
    
    def __init__(self, n_samples: int = 100000, random_state: int = 42):
        """
        Initialize data generator
        
        Args:
            n_samples: Number of loan applications to generate
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        np.random.seed(random_state)
        self.data = None
        
    def generate_data(self) -> pd.DataFrame:
        """
        Generate complete loan dataset with realistic distributions
        
        Returns:
            DataFrame with loan applications
        """
        print(f"Generating synthetic loan dataset with {self.n_samples:,} records...")
        
        # Demographics
        age = np.random.normal(loc=38, scale=12, size=self.n_samples).clip(18, 80)
        
        # Income (log-normal distribution, realistic for income)
        income = np.random.lognormal(mean=10.8, sigma=0.5, size=self.n_samples)
        income = income.clip(20000, 200000)  # $20k - $200k range
        
        # Employment length (correlated with age)
        employment_length = (age - 18) * np.random.uniform(0.3, 0.7, size=self.n_samples)
        employment_length = employment_length.clip(0, 40)
        
        # Credit score (normal distribution)
        credit_score = np.random.normal(loc=680, scale=80, size=self.n_samples)
        credit_score = credit_score.clip(300, 850)
        
        # Loan amount (correlated with income)
        loan_amount = income * np.random.uniform(0.3, 0.8, size=self.n_samples)
        
        # Interest rate (higher for lower credit scores)
        base_rate = 0.06  # 6% base
        credit_penalty = (750 - credit_score) / 1000  # Higher penalty for lower scores
        interest_rate = base_rate + credit_penalty.clip(0, 0.15)
        
        # Loan term (months)
        loan_term = np.random.choice([12, 24, 36, 48, 60], size=self.n_samples, 
                                     p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        # Monthly payment
        r = interest_rate / 12  # Monthly rate
        n = loan_term
        monthly_payment = loan_amount * (r * (1 + r)**n) / ((1 + r)**n - 1)
        
        # Credit utilization
        credit_limit = income * np.random.uniform(0.5, 2.0, size=self.n_samples)
        credit_used = credit_limit * np.random.beta(2, 5, size=self.n_samples)
        
        # Delinquencies (Poisson distribution)
        num_delinquencies = np.random.poisson(lam=0.5, size=self.n_samples)
        
        # Inquiries (related to credit seeking behavior)
        num_inquiries = np.random.poisson(lam=1.5, size=self.n_samples)
        
        # Existing debt (correlated with income)
        existing_debt = income * np.random.uniform(0.1, 0.6, size=self.n_samples)
        
        # Calculate default probability (logistic function of risk factors)
        debt_to_income = (loan_amount + existing_debt) / income
        payment_to_income = monthly_payment / (income / 12)
        credit_util = credit_used / credit_limit
        
        # Risk score (higher = more risky)
        risk_score = (
            5 * (debt_to_income - 0.4) +
            3 * (payment_to_income - 0.2) +
            2 * (credit_util - 0.3) +
            -0.005 * (credit_score - 680) +
            0.5 * num_delinquencies +
            0.2 * num_inquiries +
            -0.05 * employment_length
        )
        
        # Convert to probability using logistic function
        prob_default = 1 / (1 + np.exp(-risk_score))
        
        # Generate actual defaults
        default = np.random.binomial(1, prob_default)
        
        # Add some missing values (realistic)
        def add_missing(arr, missing_rate=0.02):
            mask = np.random.random(len(arr)) < missing_rate
            arr = arr.copy()
            arr[mask] = np.nan
            return arr
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'age': age,
            'income': add_missing(income, 0.01),
            'employment_length': add_missing(employment_length, 0.03),
            'credit_score': add_missing(credit_score, 0.02),
            'loan_amount': loan_amount,
            'interest_rate': interest_rate,
            'loan_term': loan_term,
            'monthly_payment': monthly_payment,
            'credit_limit': credit_limit,
            'credit_used': credit_used,
            'num_delinquencies': num_delinquencies,
            'num_inquiries': num_inquiries,
            'existing_debt': existing_debt,
            'debt_to_income': debt_to_income,
            'payment_to_income': payment_to_income,
            'default': default
        })
        
        default_rate = self.data['default'].mean() * 100
        print(f"✓ Generated {len(self.data):,} loan applications")
        print(f"  Default rate: {default_rate:.1f}%")
        print(f"  Features: {len(self.data.columns)}")
        
        return self.data
    
    def perform_eda(self, save_path: str = './visualizations/'):
        """
        Perform exploratory data analysis and create visualizations
        
        Args:
            save_path: Directory to save plots
        """
        if self.data is None:
            raise ValueError("Generate data first using generate_data()")
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print("\nPerforming Exploratory Data Analysis...")
        
        # Summary statistics
        print("\nSummary Statistics:")
        print(self.data.describe())
        
        # Missing values
        missing = self.data.isnull().sum()
        if missing.any():
            print("\nMissing Values:")
            print(missing[missing > 0])
        
        # Create visualizations
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Default rate
        ax1 = plt.subplot(3, 3, 1)
        default_counts = self.data['default'].value_counts()
        ax1.bar(['No Default', 'Default'], default_counts.values, color=['green', 'red'])
        ax1.set_title('Default Distribution')
        ax1.set_ylabel('Count')
        
        # 2. Income distribution
        ax2 = plt.subplot(3, 3, 2)
        ax2.hist(self.data['income'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax2.set_title('Income Distribution')
        ax2.set_xlabel('Annual Income ($)')
        ax2.set_ylabel('Frequency')
        
        # 3. Credit score distribution
        ax3 = plt.subplot(3, 3, 3)
        ax3.hist(self.data['credit_score'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax3.set_title('Credit Score Distribution')
        ax3.set_xlabel('Credit Score')
        
        # 4. Debt-to-Income by default status
        ax4 = plt.subplot(3, 3, 4)
        self.data.boxplot(column='debt_to_income', by='default', ax=ax4)
        ax4.set_title('Debt-to-Income Ratio by Default Status')
        ax4.set_xlabel('Default (0=No, 1=Yes)')
        ax4.set_ylabel('Debt-to-Income Ratio')
        plt.sca(ax4)
        plt.xticks([1, 2], ['No Default', 'Default'])
        
        # 5. Credit score vs default rate
        ax5 = plt.subplot(3, 3, 5)
        credit_bins = pd.cut(self.data['credit_score'], bins=10)
        default_by_credit = self.data.groupby(credit_bins)['default'].mean()
        default_by_credit.plot(kind='bar', ax=ax5, color='coral')
        ax5.set_title('Default Rate by Credit Score')
        ax5.set_xlabel('Credit Score Range')
        ax5.set_ylabel('Default Rate')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Income vs loan amount (colored by default)
        ax6 = plt.subplot(3, 3, 6)
        colors = ['green' if d == 0 else 'red' for d in self.data['default']]
        ax6.scatter(self.data['income'], self.data['loan_amount'], 
                   c=colors, alpha=0.3, s=10)
        ax6.set_title('Income vs Loan Amount')
        ax6.set_xlabel('Annual Income ($)')
        ax6.set_ylabel('Loan Amount ($)')
        
        # 7. Correlation heatmap
        ax7 = plt.subplot(3, 3, 7)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr()
        sns.heatmap(corr_matrix[['default']].sort_values(by='default', ascending=False),
                   annot=True, fmt='.2f', cmap='RdYlGn_r', center=0, ax=ax7,
                   cbar_kws={'shrink': 0.8})
        ax7.set_title('Feature Correlation with Default')
        
        # 8. Age distribution by default
        ax8 = plt.subplot(3, 3, 8)
        self.data[self.data['default']==0]['age'].hist(bins=30, alpha=0.5, 
                                                        label='No Default', ax=ax8)
        self.data[self.data['default']==1]['age'].hist(bins=30, alpha=0.5, 
                                                        label='Default', ax=ax8)
        ax8.set_title('Age Distribution by Default Status')
        ax8.set_xlabel('Age')
        ax8.set_ylabel('Frequency')
        ax8.legend()
        
        # 9. Loan term distribution
        ax9 = plt.subplot(3, 3, 9)
        term_counts = self.data['loan_term'].value_counts().sort_index()
        ax9.bar(term_counts.index, term_counts.values, color='skyblue', edgecolor='black')
        ax9.set_title('Loan Term Distribution')
        ax9.set_xlabel('Loan Term (months)')
        ax9.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}eda_visualizations.png', dpi=300, bbox_inches='tight')
        print(f"✓ Visualizations saved to {save_path}eda_visualizations.png")
        
        return self.data
    
    def split_data(self, test_size: float = 0.15, val_size: float = 0.15) -> Tuple:
        """
        Split data into train, validation, and test sets
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        if self.data is None:
            raise ValueError("Generate data first using generate_data()")
        
        # Separate features and target
        X = self.data.drop('default', axis=1)
        y = self.data['default']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"\nData Split:")
        print(f"  Training: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
        print(f"\n  Training default rate: {y_train.mean()*100:.1f}%")
        print(f"  Validation default rate: {y_val.mean()*100:.1f}%")
        print(f"  Test default rate: {y_test.mean()*100:.1f}%")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


# Example usage
if __name__ == "__main__":
    # Generate data
    generator = LoanDataGenerator(n_samples=100000)
    data = generator.generate_data()
    
    # Perform EDA
    generator.perform_eda()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = generator.split_data()
    
    print("\n" + "="*70)
    print("Data generation and EDA complete!")
    print("="*70)

