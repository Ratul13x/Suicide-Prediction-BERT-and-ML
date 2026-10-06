# Suicide Prediction BERT and ML

A machine learning project for suicide-risk text detection using natural language processing (NLP). This repository contains an end-to-end notebook that loads a labeled dataset of text messages, cleans and normalizes the text, extracts features, and compares several traditional machine learning classifiers to detect whether a message is related to suicide risk.

## Project Overview

The main notebook, `final_suicide_detection_pipeline.ipynb`, performs the following steps:

- Loads a suicide/non-suicide text dataset
- Explores the dataset with basic EDA
- Cleans and normalizes raw text
- Extracts features using TF-IDF vectorization
- Trains and compares multiple models
- Evaluates performance using metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix

This project is designed as a practical NLP classification pipeline and can serve as a starting point for more advanced transformer-based models such as BERT.

## Repository Contents

- `final_suicide_detection_pipeline.ipynb` — complete notebook containing the data cleaning, modeling, and evaluation pipeline
- `dataset link.txt` — link to the dataset used in the project
- `final_suicide_detection_pipeline.ipynb - Colab.pdf` — exported PDF version of the notebook for reference

## Dataset

The project uses the Kaggle suicide-watch dataset:

- https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch

The notebook expects a CSV file named `Suicide_Detection.csv` and reads it from a Google Drive location in the Colab environment:

```python
df = pd.read_csv('/content/drive/MyDrive/Suicide_Detection.csv', index_col=0)
```

If you are running it locally, update the path to match your dataset location.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TF-IDF vectorization
- Classical ML classifiers (Logistic Regression, Random Forest, KNN)

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Running the Notebook

1. Download the dataset from the Kaggle link above.
2. Place the CSV file in a local folder or Google Drive.
3. Open the notebook in Jupyter Notebook or Google Colab.
4. Update the file path if needed.
5. Run all cells in order.

## Model Workflow

The pipeline in the notebook follows this workflow:

1. Import required libraries
2. Load dataset
3. Explore the data
4. Clean the text
5. Build TF-IDF features
6. Split into train/test sets
7. Train multiple models
8. Compare model performance
9. Analyze the confusion matrix and classification metrics

## Example Usage

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
```

## Notes

- This project is intended for research, experimentation, and educational purposes.
- It is not a substitute for professional mental health diagnosis or intervention.
- The current implementation uses classical NLP + machine learning methods; a transformer-based BERT model could be added for improved performance on complex text patterns.

## License

This project does not currently include a formal license file. Please check with the repository owner before using the code commercially or in production environments.

## Acknowledgments

Thanks to the creators of the dataset and the open-source Python community for their tools and libraries used in this project.
