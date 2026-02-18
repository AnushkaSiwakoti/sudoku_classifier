# Sudoku Difficulty Classifier

Can you predict how hard a Sudoku puzzle is just by looking at the grid? I built a classifier that does exactly that - predicting whether puzzles are easy, medium, or hard with 99.5% accuracy.

## What This Project Does

Given a Sudoku grid, the model predicts its difficulty level. The interesting part isn't the classification itself - it's figuring out what makes a puzzle "hard" in the first place. Turns out, it's not just about counting empty cells.

I extracted 19 features from the raw puzzle grids, including things like how isolated the empty cells are and how the clues are distributed across different regions. The logistic regression model uses these features to classify puzzles into three difficulty levels.

## Project Structure

```
sudoku_classifier/
├── data/                   # Training and test datasets
├── src/                    # Source code
│   ├── generate_data.py    # Creates synthetic Sudoku puzzles
│   ├── feature_engineering.py   # Extracts features from puzzles
│   └── train_model.py      # Trains and evaluates the model
├── models/                 # Saved trained model
├── results/                # Confusion matrix, metrics, plots
├── notebooks/              # EDA notebook
└── run_pipeline.py         # Runs the whole thing
```

## Getting Started

Install the requirements:
```bash
pip install -r requirements.txt
```

Run everything at once:
```bash
python run_pipeline.py
```

This generates the data, extracts features, trains the model, and creates all the evaluation plots. Takes about 10 seconds.

If you want to run things step by step instead:
```bash
cd src
python generate_data.py
python feature_engineering.py
python train_model.py
```

To explore the data:
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## The Features

I engineered 19 features from the raw 9x9 grids. Here's what they capture:

**Basic counts:**
- How many cells are empty vs filled
- Fill ratio (what percentage of the grid is filled)

**Distribution across regions:**
For rows, columns, and 3x3 boxes, I calculated the mean, standard deviation, min, and max number of empty cells. This captures whether the difficulty is spread evenly or concentrated in certain areas.

**Advanced features:**
- **isolated_cells**: Empty cells that have lots of other empty cells around them. These require more advanced solving techniques.
- **givens_variance**: How evenly the clues are distributed across the 3x3 boxes
- **digit_entropy**: Measures how uniform the digit distribution is

The most important feature ended up being `isolated_cells` - when empty cells cluster together, puzzles get significantly harder.

## Results

The model gets 99.5% accuracy on the test set - only 1 mistake out of 200 puzzles. That one error was a hard puzzle classified as medium, which makes sense since it was probably borderline.

Performance breakdown:
- Easy puzzles: 100% accuracy (57/57)
- Hard puzzles: 98% accuracy (59/60) 
- Medium puzzles: 100% accuracy (83/83)

The features that mattered most:
1. `isolated_cells` - by far the strongest predictor
2. `empty_cells` - total count of empty cells
3. `fill_ratio` - how full the grid is
4. Regional distribution stats (mean empty cells per row/column/box)

Interestingly, the simple count features are almost as important as the more complex ones. Sometimes the obvious features really are the best ones.

## How to Use the Model

```python
from src.train_model import SudokuClassifier
import pandas as pd

# Load the trained model
classifier = SudokuClassifier.load_model('models/sudoku_classifier.pkl')

# Load your puzzles (needs to have features already extracted)
new_puzzles = pd.read_csv('new_puzzles_features.csv')

# Get predictions
X_new, _ = classifier.prepare_data(new_puzzles, is_training=False)
predictions = classifier.predict(X_new)
probabilities = classifier.predict_proba(X_new)
```

Or use the simpler prediction script:
```bash
python predict.py
```

## Technical Details

**Data:** 800 training puzzles, 200 test puzzles. Synthetic but realistic - I generated them with controlled difficulty levels based on how many cells are empty.

**Model:** Logistic regression with LBFGS solver. Simple baseline that worked surprisingly well - no need for random forests or neural networks when you have good features.

**Preprocessing:** Standard scaling and label encoding. Fit the scaler on training data only to avoid data leakage.

**Train/test split:** 80/20 split with fixed random seed for reproducibility.

## What I'd Do Differently

If I were to keep working on this:

- **Try other models** - Random Forest or XGBoost might pick up on feature interactions better
- **Add cross-validation** - Right now I'm just using one train/test split
- **Use real puzzles** - These are synthetic. Real competition puzzles would be more interesting
- **More granular difficulty** - Instead of just easy/medium/hard, maybe rate on a 1-10 scale
- **Better features** - Could add things like "constraint propagation depth" or detecting naked pairs/triples
- **Deploy it** - Make a simple web app where you upload a puzzle and get the difficulty prediction

## What I Learned

The biggest lesson: **feature engineering matters more than model complexity**. I spent way more time thinking about what makes puzzles hard than I did tuning hyperparameters, and that's where the value came from.

Also learned that the `isolated_cells` feature - which I designed based on understanding how Sudoku solving works - ended up being the most important predictor. Domain knowledge really does make a difference.
