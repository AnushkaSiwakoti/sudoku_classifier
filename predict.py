"""
Make predictions using trained Sudoku Classifier.
Example script showing how to use the trained model.
"""
import numpy as np
import pandas as pd
from src.train_model import SudokuClassifier
from src.feature_engineering import SudokuFeatureExtractor


def predict_single_puzzle(puzzle: np.ndarray, model_path: str = 'models/sudoku_classifier.pkl'):
    """
    Predict difficulty of a single Sudoku puzzle.
    
    Parameters:
    -----------
    puzzle : np.ndarray
        9x9 Sudoku grid (0 for empty cells)
    model_path : str
        Path to trained model
    
    Returns:
    --------
    dict with prediction and probabilities
    """
    # Load model
    classifier = SudokuClassifier.load_model(model_path)
    
    # Extract features
    extractor = SudokuFeatureExtractor()
    features = extractor.extract_features(puzzle)
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features])
    
    # Prepare data (transform only, not fit)
    X, _ = classifier.prepare_data(features_df, is_training=False)
    
    # Make prediction
    prediction = classifier.predict(X)[0]
    probabilities = classifier.predict_proba(X)[0]
    
    # Get class names
    difficulty = classifier.label_encoder.inverse_transform([prediction])[0]
    class_probs = {
        cls: prob 
        for cls, prob in zip(classifier.label_encoder.classes_, probabilities)
    }
    
    return {
        'difficulty': difficulty,
        'probabilities': class_probs,
        'features': features
    }


def example_predictions():
    """Run example predictions."""
    print("="*70)
    print("  Sudoku Difficulty Classifier - Prediction Examples")
    print("="*70)
    
    # Example 1: Easy puzzle (few empty cells)
    easy_puzzle = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 0, 0],  # Only 2 empty cells
        [3, 4, 5, 2, 8, 6, 0, 0, 0]   # 3 empty cells
    ])
    
    # Example 2: Hard puzzle (many empty cells)
    hard_puzzle = np.array([
        [0, 0, 0, 0, 0, 0, 9, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 8],
        [0, 0, 8, 0, 0, 2, 0, 6, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 9, 0, 0, 0, 0, 6],
        [0, 0, 0, 5, 0, 0, 0, 0, 0],
        [0, 8, 0, 4, 0, 0, 0, 0, 0],
        [0, 4, 0, 2, 0, 6, 0, 0, 0]
    ])
    
    puzzles = [
        ('Easy Puzzle Example', easy_puzzle),
        ('Hard Puzzle Example', hard_puzzle)
    ]
    
    for name, puzzle in puzzles:
        print(f"\n{name}")
        print("-"*70)
        print("Puzzle:")
        print(puzzle)
        print(f"\nEmpty cells: {np.sum(puzzle == 0)}")
        
        # Make prediction
        result = predict_single_puzzle(puzzle)
        
        print(f"\n✓ Predicted Difficulty: {result['difficulty'].upper()}")
        print("\nClass Probabilities:")
        for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls:8s}: {prob:.2%} {'█' * int(prob * 50)}")
        
        print("\nKey Features:")
        key_features = ['empty_cells', 'isolated_cells', 'fill_ratio', 'digit_entropy']
        for feat in key_features:
            if feat in result['features']:
                print(f"  {feat:20s}: {result['features'][feat]:.3f}")


def predict_from_csv(csv_path: str, model_path: str = 'models/sudoku_classifier.pkl'):
    """
    Make predictions for puzzles in a CSV file.
    
    CSV should have columns 'cell_0' through 'cell_80' (81 cells in row-major order).
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file with puzzles
    model_path : str
        Path to trained model
    
    Returns:
    --------
    pd.DataFrame with predictions
    """
    # Load model
    classifier = SudokuClassifier.load_model(model_path)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Extract features
    extractor = SudokuFeatureExtractor()
    features_df = extractor.transform_dataset(df)
    
    # Remove difficulty column if present
    if 'difficulty' in features_df.columns:
        X_df = features_df.drop('difficulty', axis=1)
    else:
        X_df = features_df
    
    # Prepare data
    X, _ = classifier.prepare_data(X_df, is_training=False)
    
    # Make predictions
    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)
    
    # Convert to class names
    difficulty_labels = classifier.label_encoder.inverse_transform(predictions)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'puzzle_id': range(len(predictions)),
        'predicted_difficulty': difficulty_labels,
        'prob_easy': probabilities[:, classifier.label_encoder.transform(['easy'])[0]],
        'prob_hard': probabilities[:, classifier.label_encoder.transform(['hard'])[0]],
        'prob_medium': probabilities[:, classifier.label_encoder.transform(['medium'])[0]]
    })
    
    return results


if __name__ == '__main__':
    # Run example predictions
    example_predictions()
    
    print("\n" + "="*70)
    print("  To use this model in your own code:")
    print("="*70)
    print("""
from predict import predict_single_puzzle
import numpy as np

# Your 9x9 Sudoku puzzle (0 for empty cells)
my_puzzle = np.array([
    [0, 0, 0, ...],
    ...
])

# Get prediction
result = predict_single_puzzle(my_puzzle)
print(f"Difficulty: {result['difficulty']}")
print(f"Probabilities: {result['probabilities']}")
    """)
