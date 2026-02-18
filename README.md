# 🎯 Sudoku Difficulty Classifier

A machine learning project that classifies Sudoku puzzles by difficulty level (easy, medium, hard) using logistic regression. This project demonstrates core ML fundamentals including feature engineering, model training, and evaluation.

## 📋 Project Overview

This project builds a classifier to predict the difficulty of Sudoku puzzles based on their structural characteristics. It showcases:

- **Feature Engineering**: Extracting meaningful features from raw puzzle grids
- **ML Pipeline**: Clean, reproducible workflow from data to predictions
- **Model Evaluation**: Comprehensive metrics and visualization
- **Best Practices**: Proper train/test split, preprocessing, and evaluation discipline

## 🏗️ Project Structure

```
sudoku_classifier/
├── data/                          # Data files
│   ├── train.csv                  # Raw training data (generated)
│   ├── test.csv                   # Raw test data (generated)
│   ├── train_features.csv         # Engineered training features
│   └── test_features.csv          # Engineered test features
├── notebooks/                     # Jupyter notebooks
│   └── 01_exploratory_data_analysis.ipynb
├── src/                          # Source code
│   ├── generate_data.py          # Data generation script
│   ├── feature_engineering.py    # Feature extraction module
│   └── train_model.py            # Model training pipeline
├── models/                       # Saved models
│   └── sudoku_classifier.pkl     # Trained classifier
├── results/                      # Evaluation results
│   ├── confusion_matrix.png      # Confusion matrix plot
│   ├── feature_importance.png    # Feature importance plot
│   └── metrics.csv               # Performance metrics
├── requirements.txt              # Python dependencies
├── run_pipeline.py              # Execute full pipeline
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd sudoku_classifier

# Install dependencies
pip install -r requirements.txt --break-system-packages
```

### Run the Complete Pipeline

```bash
# Execute full pipeline (data generation → training → evaluation)
python run_pipeline.py
```

Or run each step individually:

```bash
# 1. Generate data
cd src
python generate_data.py

# 2. Extract features
python feature_engineering.py

# 3. Train model and evaluate
python train_model.py
```

### Explore the Data

```bash
# Launch Jupyter notebook for EDA
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

## 🔧 Features Engineered

The classifier uses 19 engineered features extracted from raw Sudoku grids:

### Basic Features
- **empty_cells**: Number of unfilled cells
- **filled_cells**: Number of given cells
- **fill_ratio**: Proportion of filled cells

### Row Statistics
- **empty_per_row_mean**: Average empty cells per row
- **empty_per_row_std**: Standard deviation of empty cells per row
- **empty_per_row_min**: Minimum empty cells in any row
- **empty_per_row_max**: Maximum empty cells in any row

### Column Statistics
- **empty_per_col_mean**: Average empty cells per column
- **empty_per_col_std**: Standard deviation of empty cells per column
- **empty_per_col_min**: Minimum empty cells in any column
- **empty_per_col_max**: Maximum empty cells in any column

### Box Statistics (3x3 regions)
- **empty_per_box_mean**: Average empty cells per box
- **empty_per_box_std**: Standard deviation of empty cells per box
- **empty_per_box_min**: Minimum empty cells in any box
- **empty_per_box_max**: Maximum empty cells in any box

### Advanced Features
- **isolated_cells**: Count of empty cells with many empty neighbors
- **givens_variance**: Variance in number of given cells per box
- **digit_entropy**: Entropy of digit distribution (uniformity measure)

## 📊 Model Performance

The logistic regression baseline achieves strong performance:

- **Accuracy**: ~95%+ on test set
- **Macro F1-Score**: ~95%+
- **Balanced Performance**: High precision and recall across all difficulty levels

Key insights:
- `empty_cells` is the strongest single predictor
- Variance features (`empty_per_row_std`, `empty_per_box_std`) add significant value
- `isolated_cells` helps distinguish hard puzzles
- Clean separation between difficulty classes

## 🎓 Skills Demonstrated

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis
- Feature visualization
- Correlation analysis
- Outlier detection

### 2. Feature Engineering
- Domain knowledge application
- Statistical feature extraction
- Feature selection based on separability

### 3. ML Fundamentals
- Train/test split discipline (800/200 split)
- Data preprocessing (standardization, label encoding)
- Logistic regression for multi-class classification
- Hyperparameter awareness

### 4. Model Evaluation
- Multiple metrics: accuracy, precision, recall, F1
- Confusion matrix analysis
- Feature importance interpretation
- Proper evaluation on held-out test set

### 5. Clean Code Practices
- Modular design
- Reproducible pipeline
- Documentation
- Version control friendly structure

## 📈 Example Usage

```python
from src.train_model import SudokuClassifier
import pandas as pd

# Load trained model
classifier = SudokuClassifier.load_model('models/sudoku_classifier.pkl')

# Load new puzzle data (with features extracted)
new_puzzles = pd.read_csv('new_puzzles_features.csv')

# Prepare data
X_new, _ = classifier.prepare_data(new_puzzles, is_training=False)

# Make predictions
predictions = classifier.predict(X_new)
probabilities = classifier.predict_proba(X_new)

# predictions will be: array([0, 2, 1, ...])  # 0=easy, 1=hard, 2=medium
# probabilities will be: array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], ...])
```

## 🔍 Project Highlights

### Data Generation
- Synthetic but realistic Sudoku puzzles
- Controlled difficulty levels based on number of empty cells
- Balanced class distribution

### Feature Engineering
- 19 carefully designed features
- Mix of basic counts, statistical measures, and structural properties
- Features capture puzzle difficulty from multiple angles

### ML Pipeline
- Clean separation of concerns
- Proper data preprocessing
- Reproducible results (fixed random seed)
- Model persistence for deployment

### Evaluation
- Comprehensive metrics
- Visual analysis (confusion matrix, feature importance)
- Comparison of train vs test performance
- Feature importance ranking

## 🛠️ Technical Details

### Model
- **Algorithm**: Logistic Regression (multinomial)
- **Solver**: LBFGS
- **Max Iterations**: 1000
- **Multi-class**: One-vs-Rest approach

### Preprocessing
- **Scaling**: StandardScaler (zero mean, unit variance)
- **Label Encoding**: easy=0, hard=1, medium=2

### Data Split
- **Training**: 800 samples (80%)
- **Testing**: 200 samples (20%)
- **No cross-validation** (baseline model)

## 📝 Future Improvements

1. **Model Enhancement**
   - Try other algorithms (Random Forest, SVM, Neural Networks)
   - Implement cross-validation
   - Hyperparameter tuning

2. **Feature Engineering**
   - Add constraint propagation depth features
   - Include naked pairs/triples detection
   - Solver-based difficulty metrics

3. **Data**
   - Use real Sudoku puzzles from competitions
   - Increase dataset size
   - Balance by true solving difficulty

4. **Deployment**
   - Create web API
   - Build interactive demo
   - Mobile app integration

## 📚 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Feature Engineering Techniques](https://www.kaggle.com/learn/feature-engineering)
- [Logistic Regression Theory](https://en.wikipedia.org/wiki/Logistic_regression)
- [Sudoku Solving Algorithms](https://en.wikipedia.org/wiki/Sudoku_solving_algorithms)

## 🤝 Contributing

This is a portfolio/learning project, but feedback and suggestions are welcome!

## 📄 License

MIT License - feel free to use this code for learning and projects.

---

**Built with**: Python 3.8+, scikit-learn, pandas, matplotlib, seaborn

**Author**: ML Practitioner focusing on fundamentals and best practices
