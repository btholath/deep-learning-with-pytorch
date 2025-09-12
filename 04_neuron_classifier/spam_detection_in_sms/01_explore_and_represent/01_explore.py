"""
Use dataset at /workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection
This is a classic dataset used for binary classification:
    Task → classify each SMS message as either spam or ham (not spam).
Binary classification means we only have two classes (1 = spam, 0 = ham).
"""
# Load and explore the dataset
import pandas as pd

# Load dataset
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection", sep="\t", names=["label", "message"])

# Peek at the data
print(df.head())
print(df["label"].value_counts())

# Convert labels to numbers (binary target)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
print(df["label_num"] )

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label_num"], test_size=0.2, random_state=42
)

# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# STEP: 5
# Train the model: here using Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)


# STEP: 6
# Evaluate using accuracy, precision, recall, F1-score
# Predict
y_pred = model.predict(X_test_vec)

# Evaluate
print(classification_report(y_test, y_pred))
