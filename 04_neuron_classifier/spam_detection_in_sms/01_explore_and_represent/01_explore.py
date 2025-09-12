"""
GOAL (in simple words):
Look inside the SMS dataset and see what we're dealing with.

WHY THIS MATTERS:
Before teaching a computer to detect spam, we must understand what spam/ham messages look like and how many of each we have.

Use dataset at /workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection




This is a classic dataset used for binary classification:
    Task → classify each SMS message as either spam or ham (not spam).
Binary classification means we only have two classes (1 = spam, 0 = ham).
"""
# Load and explore the dataset
import pandas as pd

# Load dataset
# 1) Load the data
# The dataset is a text file with 2 columns per line, separated by a tab:
#   - Column 1: "ham" or "spam"
#   - Column 2: the SMS text
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection", sep="\t", names=["label", "message"])


# 2) Peek at a few rows to get a feel for the messages
print("First 5 rows:\n", df.head(), "\n")
print(df.head())
print(df["label"].value_counts())

# 3) Count how many spam vs ham messages we have
print("Counts:\n", df["label"].value_counts(), "\n")
# Convert labels to numbers (binary target)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
print(df["label_num"] )

# 4) Show a few examples of each (so students can see the difference)
print("Sample HAM messages:\n", df[df["label"] == "ham"]["message"].head(3).tolist(), "\n")
print("Sample SPAM messages:\n", df[df["label"] == "spam"]["message"].head(3).tolist(), "\n")
