"""
Feature engineering for Sudoku difficulty classification.
Extract meaningful features from raw puzzle grids.
"""
import numpy as np
import pandas as pd
from typing import List, Dict


class SudokuFeatureExtractor:
    """Extract features from Sudoku puzzles."""
    
    def __init__(self):
        self.feature_names = []
    
    def count_empty_cells(self, puzzle: np.ndarray) -> int:
        """Count number of empty cells (zeros)."""
        return np.sum(puzzle == 0)
    
    def count_filled_cells(self, puzzle: np.ndarray) -> int:
        """Count number of filled cells."""
        return np.sum(puzzle != 0)
    
    def empty_cells_per_row(self, puzzle: np.ndarray) -> List[float]:
        """Statistics of empty cells per row."""
        empty_per_row = np.sum(puzzle == 0, axis=1)
        return [
            np.mean(empty_per_row),
            np.std(empty_per_row),
            np.min(empty_per_row),
            np.max(empty_per_row)
        ]
    
    def empty_cells_per_col(self, puzzle: np.ndarray) -> List[float]:
        """Statistics of empty cells per column."""
        empty_per_col = np.sum(puzzle == 0, axis=0)
        return [
            np.mean(empty_per_col),
            np.std(empty_per_col),
            np.min(empty_per_col),
            np.max(empty_per_col)
        ]
    
    def empty_cells_per_box(self, puzzle: np.ndarray) -> List[float]:
        """Statistics of empty cells per 3x3 box."""
        empty_counts = []
        for box_row in range(3):
            for box_col in range(3):
                box = puzzle[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
                empty_counts.append(np.sum(box == 0))
        
        empty_counts = np.array(empty_counts)
        return [
            np.mean(empty_counts),
            np.std(empty_counts),
            np.min(empty_counts),
            np.max(empty_counts)
        ]
    
    def count_isolated_cells(self, puzzle: np.ndarray) -> int:
        """
        Count cells that are isolated (empty with many empty neighbors).
        This makes puzzles harder.
        """
        isolated = 0
        for i in range(9):
            for j in range(9):
                if puzzle[i, j] == 0:  # Empty cell
                    # Count empty neighbors in same row/col
                    empty_neighbors = 0
                    # Check row
                    empty_neighbors += np.sum(puzzle[i, :] == 0) - 1  # Exclude self
                    # Check column
                    empty_neighbors += np.sum(puzzle[:, j] == 0) - 1  # Exclude self
                    
                    # If more than half neighbors are empty, it's isolated
                    if empty_neighbors > 9:
                        isolated += 1
        return isolated
    
    def count_givens_in_boxes(self, puzzle: np.ndarray) -> float:
        """
        Variance in number of given cells per box.
        High variance can indicate harder puzzles.
        """
        given_counts = []
        for box_row in range(3):
            for box_col in range(3):
                box = puzzle[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
                given_counts.append(np.sum(box != 0))
        
        return np.var(given_counts)
    
    def digit_distribution_entropy(self, puzzle: np.ndarray) -> float:
        """
        Entropy of digit distribution (excluding zeros).
        More uniform distribution might indicate harder puzzles.
        """
        non_zero = puzzle[puzzle != 0].astype(int)
        if len(non_zero) == 0:
            return 0
        
        # Count frequency of each digit
        counts = np.bincount(non_zero, minlength=10)[1:]  # Exclude 0
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]  # Remove zero probabilities
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        return entropy
    
    def extract_features(self, puzzle: np.ndarray) -> Dict[str, float]:
        """Extract all features from a puzzle."""
        features = {}
        
        # Basic counts
        features['empty_cells'] = self.count_empty_cells(puzzle)
        features['filled_cells'] = self.count_filled_cells(puzzle)
        features['fill_ratio'] = features['filled_cells'] / 81.0
        
        # Row statistics
        row_stats = self.empty_cells_per_row(puzzle)
        features['empty_per_row_mean'] = row_stats[0]
        features['empty_per_row_std'] = row_stats[1]
        features['empty_per_row_min'] = row_stats[2]
        features['empty_per_row_max'] = row_stats[3]
        
        # Column statistics
        col_stats = self.empty_cells_per_col(puzzle)
        features['empty_per_col_mean'] = col_stats[0]
        features['empty_per_col_std'] = col_stats[1]
        features['empty_per_col_min'] = col_stats[2]
        features['empty_per_col_max'] = col_stats[3]
        
        # Box statistics
        box_stats = self.empty_cells_per_box(puzzle)
        features['empty_per_box_mean'] = box_stats[0]
        features['empty_per_box_std'] = box_stats[1]
        features['empty_per_box_min'] = box_stats[2]
        features['empty_per_box_max'] = box_stats[3]
        
        # Advanced features
        features['isolated_cells'] = self.count_isolated_cells(puzzle)
        features['givens_variance'] = self.count_givens_in_boxes(puzzle)
        features['digit_entropy'] = self.digit_distribution_entropy(puzzle)
        
        return features
    
    def transform_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataset from raw cells to engineered features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with columns 'cell_0' to 'cell_80' and 'difficulty'
        
        Returns:
        --------
        pd.DataFrame with engineered features and difficulty label
        """
        features_list = []
        
        for idx, row in df.iterrows():
            # Reconstruct puzzle
            puzzle = row[[f'cell_{i}' for i in range(81)]].values.astype(int).reshape(9, 9)
            
            # Extract features
            features = self.extract_features(puzzle)
            features['difficulty'] = row['difficulty']
            features_list.append(features)
        
        return pd.DataFrame(features_list)


def main():
    """Transform raw data to features."""
    import os
    
    # Load raw data
    print("Loading raw data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    # Extract features
    print("Extracting features...")
    extractor = SudokuFeatureExtractor()
    
    train_features = extractor.transform_dataset(train_df)
    test_features = extractor.transform_dataset(test_df)
    
    # Save feature datasets
    print("Saving feature datasets...")
    train_features.to_csv('data/train_features.csv', index=False)
    test_features.to_csv('data/test_features.csv', index=False)
    
    print(f"\nTraining features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    print(f"\nFeatures extracted: {list(train_features.columns[:-1])}")


if __name__ == '__main__':
    main()
