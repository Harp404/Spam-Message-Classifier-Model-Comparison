# Spam Message Classifier - Model Comparison

A machine learning project that trains and compares multiple algorithms to classify messages as spam or ham (not spam). The script automatically selects the best performing model and saves it for future use.

## Features

- **Multiple Model Comparison**: Tests 4 different machine learning algorithms
  - Logistic Regression
  - Naive Bayes
  - XGBoost
  - Random Forest
- **Automatic Best Model Selection**: Chooses the highest accuracy model
- **Text Preprocessing**: Uses TF-IDF vectorization for feature extraction
- **Model Persistence**: Saves the best model and tokenizer for reuse
- **Interactive Testing**: Test the trained model with custom messages

## Project Structure

```
Spam-Message-Classifier-model-comparison/
├── model.py                              # Main training and comparison script
├── data.csv                              # Training dataset
├── model_[BestModel].pkl                 # Saved best model (generated)
├── model_tokenizer_[BestModel].pkl       # Saved tokenizer (generated)
└── README.md                             # Project documentation
```

## Dataset Format

The CSV file should contain two columns:
- `Category`: Either "spam" or "ham"
- `Message`: The text message to classify

Example:
```csv
Category,Message
ham,"Go until jurong point, crazy.. Available only..."
spam,"Free entry in 2 a wkly comp to win FA Cup final..."
```

## Installation

Make sure you have Python installed with the following packages:

```bash
pip install pandas scikit-learn xgboost pickle
```

## Usage

1. **Prepare your dataset**: Ensure your CSV file has 'Category' and 'Message' columns
2. **Run the training script**:
   ```bash
   python model.py
   ```
3. **Follow the prompts**:
   - Enter the path to your CSV file (or press Enter for default `data.csv`)
   - Enter a name for your model (or press Enter for default naming)
4. **View results**: The script will display accuracy comparisons and save the best model
5. **Test interactively**: Enter messages to classify them in real-time

## How It Works

1. **Data Loading**: Reads the CSV file and prepares the dataset
2. **Text Preprocessing**: Converts text messages to numerical features using TF-IDF vectorization (5000 features)
3. **Data Splitting**: Splits data into 80% training and 20% testing
4. **Model Training**: Trains all four algorithms on the training data
5. **Evaluation**: Tests each model on the test set and calculates accuracy
6. **Best Model Selection**: Automatically selects the model with highest accuracy
7. **Model Saving**: Saves both the best model and tokenizer as pickle files
8. **Interactive Testing**: Allows real-time classification of new messages

## Model Details

- **Logistic Regression**: L2 penalty, 500 max iterations
- **Naive Bayes**: Multinomial with alpha=0.5
- **XGBoost**: Default XGBoost classifier
- **Random Forest**: Default Random Forest classifier

## Output Files

After running the script, you'll get:
- `model_[BestModelName].pkl`: The trained best performing model
- `model_tokenizer_[BestModelName].pkl`: The TF-IDF tokenizer used for preprocessing

## Example Output

```
================================================================================
SPAM MESSAGE CLASSIFIER - MODEL COMPARISON & TRAINING
================================================================================
...
==================================================
MODEL PERFORMANCE COMPARISON
==================================================
Logistic Regression: 95.23%
Naive Bayes:         94.87%
XGBoost:             96.12%
Random Forest:       95.45%
==================================================
🏆 BEST MODEL: XGBoost (Accuracy: 96.12%)
```

## Interactive Testing

After training, you can test the model with custom messages:

```
Enter the message to classify (q to quit): Free lottery winner! Claim now!
🚨 Classification: SPAM

Enter the message to classify (q to quit): Hey, are we still meeting today?
✅ Classification: HAM (Not Spam)
```

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- xgboost
- pickle (built-in)
- warnings (built-in)
- os (built-in)

## Notes

- The script uses a random state of 42 for reproducible results
- TF-IDF vectorization is limited to 5000 features for optimal performance
- All warnings are suppressed for cleaner output
- The model automatically maps 'ham' to 0 and 'spam' to 1 for binary classification