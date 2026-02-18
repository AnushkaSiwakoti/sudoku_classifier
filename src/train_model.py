"""
ML Pipeline for Sudoku Difficulty Classification.
Implements logistic regression baseline with proper train/test discipline.
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class SudokuClassifier:
    """
    Sudoku difficulty classifier using Logistic Regression.
    Follows ML best practices with proper preprocessing and evaluation.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs'
        )
        self.feature_names = None
        self.is_fitted = False
    
    def prepare_data(self, df: pd.DataFrame, is_training: bool = True):
        """
        Prepare features and labels from dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features and 'difficulty' column
        is_training : bool
            If True, fit scaler and label encoder. If False, transform only.
        
        Returns:
        --------
        X : np.ndarray
            Scaled features
        y : np.ndarray
            Encoded labels (None if 'difficulty' not in df)
        """
        # Separate features and labels
        if 'difficulty' in df.columns:
            X = df.drop('difficulty', axis=1).values
            y_raw = df['difficulty'].values
            
            if is_training:
                # Fit and transform
                self.feature_names = df.drop('difficulty', axis=1).columns.tolist()
                X = self.scaler.fit_transform(X)
                y = self.label_encoder.fit_transform(y_raw)
            else:
                # Transform only
                X = self.scaler.transform(X)
                y = self.label_encoder.transform(y_raw)
            
            return X, y
        else:
            X = df.values
            if is_training:
                self.feature_names = df.columns.tolist()
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
            return X, None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the logistic regression model."""
        print("Training logistic regression model...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print("Model training completed!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions!")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions!")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Evaluate model performance.
        
        Returns:
        --------
        dict with metrics: accuracy, precision, recall, f1_score
        """
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
        }
        
        return metrics
    
    def get_classification_report(self, X: np.ndarray, y_true: np.ndarray) -> str:
        """Get detailed classification report."""
        y_pred = self.predict(X)
        target_names = self.label_encoder.classes_
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def get_confusion_matrix(self, X: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Get confusion matrix."""
        y_pred = self.predict(X)
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, X: np.ndarray, y_true: np.ndarray, 
                             save_path: str = None):
        """Plot confusion matrix."""
        cm = self.get_confusion_matrix(X, y_true)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from logistic regression coefficients.
        
        Returns:
        --------
        pd.DataFrame with features and their average absolute coefficient
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first!")
        
        # Average absolute coefficient across all classes
        coeffs = np.abs(self.model.coef_).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': coeffs
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 10, save_path: str = None):
        """Plot top N most important features."""
        importance_df = self.get_feature_importance().head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_n), importance_df['importance'].values)
        plt.yticks(range(top_n), importance_df['feature'].values)
        plt.xlabel('Average Absolute Coefficient')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        plt.close()
    
    def save_model(self, path: str):
        """Save trained model and preprocessing objects."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str):
        """Load trained model."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls()
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.label_encoder = model_data['label_encoder']
        classifier.feature_names = model_data['feature_names']
        classifier.is_fitted = model_data['is_fitted']
        
        return classifier


def main():
    """Main training and evaluation pipeline."""
    import os
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load feature data
    print("Loading feature data...")
    train_df = pd.read_csv('data/train_features.csv')
    test_df = pd.read_csv('data/test_features.csv')
    
    print(f"Training set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    
    # Initialize classifier
    classifier = SudokuClassifier(random_state=42)
    
    # Prepare data
    print("\nPreparing training data...")
    X_train, y_train = classifier.prepare_data(train_df, is_training=True)
    
    print("Preparing test data...")
    X_test, y_test = classifier.prepare_data(test_df, is_training=False)
    
    # Train model
    print("\n" + "="*50)
    classifier.train(X_train, y_train)
    print("="*50)
    
    # Evaluate on training set
    print("\n📊 TRAINING SET PERFORMANCE:")
    print("-"*50)
    train_metrics = classifier.evaluate(X_train, y_train)
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTraining Classification Report:")
    print(classifier.get_classification_report(X_train, y_train))
    
    # Evaluate on test set
    print("\n📊 TEST SET PERFORMANCE:")
    print("-"*50)
    test_metrics = classifier.evaluate(X_test, y_test)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTest Classification Report:")
    print(classifier.get_classification_report(X_test, y_test))
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    classifier.plot_confusion_matrix(X_test, y_test, 
                                    save_path='results/confusion_matrix.png')
    
    # Plot feature importance
    print("Generating feature importance plot...")
    classifier.plot_feature_importance(top_n=15, 
                                      save_path='results/feature_importance.png')
    
    # Print top features
    print("\n🎯 TOP 10 MOST IMPORTANT FEATURES:")
    print("-"*50)
    importance_df = classifier.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))
    
    # Save model
    print("\nSaving trained model...")
    classifier.save_model('models/sudoku_classifier.pkl')
    
    # Save metrics
    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    
    results_df = pd.DataFrame([
        {'split': 'train', **train_metrics},
        {'split': 'test', **test_metrics}
    ])
    results_df.to_csv('results/metrics.csv', index=False)
    print("Results saved to results/metrics.csv")
    
    print("\n✅ Pipeline completed successfully!")


if __name__ == '__main__':
    main()
