"""
Generate Sudoku puzzles with difficulty labels.
Features are based on puzzle characteristics that correlate with difficulty.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List


class SudokuGenerator:
    """Generate Sudoku puzzles with varying difficulty levels."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def generate_solved_sudoku(self) -> np.ndarray:
        """Generate a valid solved 9x9 Sudoku grid."""
        # Start with a base pattern
        base = np.arange(1, 10)
        grid = np.zeros((9, 9), dtype=int)
        
        # Fill first row randomly
        grid[0] = np.random.permutation(base)
        
        # Use simple shifting pattern for valid solution
        for i in range(1, 9):
            shift = (i % 3) * 3 + i // 3
            grid[i] = np.roll(grid[0], shift)
        
        return grid
    
    def remove_cells(self, grid: np.ndarray, num_cells: int) -> np.ndarray:
        """Remove cells from a solved grid to create a puzzle."""
        puzzle = grid.copy()
        positions = [(i, j) for i in range(9) for j in range(9)]
        remove_positions = np.random.choice(len(positions), num_cells, replace=False)
        
        for idx in remove_positions:
            i, j = positions[idx]
            puzzle[i, j] = 0
        
        return puzzle
    
    def generate_puzzle(self, difficulty: str) -> np.ndarray:
        """
        Generate a puzzle with specified difficulty.
        
        Difficulty levels:
        - easy: 30-40 cells removed
        - medium: 41-52 cells removed
        - hard: 53-64 cells removed
        """
        solved = self.generate_solved_sudoku()
        
        if difficulty == 'easy':
            num_remove = np.random.randint(30, 41)
        elif difficulty == 'medium':
            num_remove = np.random.randint(41, 53)
        else:  # hard
            num_remove = np.random.randint(53, 65)
        
        return self.remove_cells(solved, num_remove)
    
    def generate_dataset(self, 
                        n_samples: int = 1000,
                        difficulty_dist: dict = None) -> pd.DataFrame:
        """
        Generate a dataset of Sudoku puzzles.
        
        Parameters:
        -----------
        n_samples : int
            Total number of puzzles to generate
        difficulty_dist : dict
            Distribution of difficulties {'easy': 0.33, 'medium': 0.33, 'hard': 0.34}
        
        Returns:
        --------
        pd.DataFrame with columns for puzzle cells and difficulty label
        """
        if difficulty_dist is None:
            difficulty_dist = {'easy': 0.33, 'medium': 0.33, 'hard': 0.34}
        
        data = []
        difficulties = list(difficulty_dist.keys())
        probs = list(difficulty_dist.values())
        
        for _ in range(n_samples):
            difficulty = np.random.choice(difficulties, p=probs)
            puzzle = self.generate_puzzle(difficulty)
            
            # Flatten puzzle to 1D array
            row_data = puzzle.flatten().tolist()
            row_data.append(difficulty)
            data.append(row_data)
        
        # Create column names
        cols = [f'cell_{i}' for i in range(81)] + ['difficulty']
        df = pd.DataFrame(data, columns=cols)
        
        return df


def main():
    """Generate and save Sudoku dataset."""
    generator = SudokuGenerator(seed=42)
    
    # Generate training data
    print("Generating training data...")
    train_df = generator.generate_dataset(n_samples=800)
    train_df.to_csv('data/train.csv', index=False)
    print(f"Training data saved: {len(train_df)} samples")
    
    # Generate test data
    print("Generating test data...")
    test_df = generator.generate_dataset(n_samples=200)
    test_df.to_csv('data/test.csv', index=False)
    print(f"Test data saved: {len(test_df)} samples")
    
    # Print distribution
    print("\nTraining set distribution:")
    print(train_df['difficulty'].value_counts())
    print("\nTest set distribution:")
    print(test_df['difficulty'].value_counts())


if __name__ == '__main__':
    main()
