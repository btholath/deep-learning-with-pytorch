"""
GOAL:
Use a modern language model (BART) to turn SMS messages into number lists
(embeddings) that capture meaning, then train a tiny classifier to tell
SPAM (1) from HAM / not-spam (0).

WHY THIS IS USEFUL:
- Email and messaging apps use models like this to filter spam.
- Embeddings help computers understand meaning, not just exact words.
"""

import sys
import torch
from torch import nn
import pandas as pd
from transformers import BartTokenizer, BartModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------------
# 1) Load and label the dataset
# -----------------------------
# Each row has:
#   - a label ("ham" or "spam")
#   - the SMS message text
df = pd.read_csv(
    "/workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection",
    sep="\t",
    names=["type", "message"]
)

# Make a numeric label:
#   spam -> 1, ham -> 0  (easier for the model)
df["spam"] = (df["type"] == "spam").astype(int)
df.drop("type", axis=1, inplace=True)  # we don't need the text label now

# -----------------------------
# 2) Split into Train / Validation sets
# -----------------------------
# Train = the messages the model learns from (80%)
# Validation = new messages (20%) to test if learning is real (not memorized)
df_train, df_val = train_test_split(df, test_size=0.2, random_state=0)

# -----------------------------
# 3) Load a pretrained "text brain" (BART)
# -----------------------------
# Tokenizer: breaks text into tokens (pieces) and turns them into numbers (IDs)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# BART model: a big neural network already trained on huge text libraries
# from_pretrained loads those learned weights for immediate use
model_bart = BartModel.from_pretrained("facebook/bart-base")

# -----------------------------
# 4) A helper to tokenize a batch of texts
# -----------------------------
# It returns PyTorch tensors BART understands: input_ids and attention_mask
def tokenize_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,       # pad shorter messages so all are same length
        truncation=True,    # cut off very long ones (keeps compute reasonable)
        return_tensors="pt",
        max_length=512
    )

# -----------------------------
# 5) Turn TRAIN messages into embeddings
# -----------------------------
print("Tokenizing and transforming training data...")
tokens_train = tokenize_texts(df_train["message"])

embeddings_train = []  # we'll collect one embedding per message
model_bart.eval()      # evaluation mode (we're not training BART)
with torch.no_grad():  # don't track gradients (faster, less memory)
    # We iterate one message at a time for clarity
    for i in tqdm(range(len(tokens_train["input_ids"]))):
        # Select the i-th message (unsqueeze adds the batch dimension)
        output = model_bart(
            input_ids=tokens_train["input_ids"][i].unsqueeze(0),
            attention_mask=tokens_train["attention_mask"][i].unsqueeze(0)
        )
        # output.last_hidden_state: [batch, seq_len, hidden_size]
        # Mean over tokens -> one vector per message (captures overall meaning)
        embedding = output.last_hidden_state.mean(dim=1).squeeze(0)
        embeddings_train.append(embedding)

# -----------------------------
# 6) Turn VAL messages into embeddings (same steps)
# -----------------------------
print("Tokenizing and transforming validation data...")
tokens_val = tokenize_texts(df_val["message"])

embeddings_val = []
model_bart.eval()
with torch.no_grad():
    for i in tqdm(range(len(tokens_val["input_ids"]))):
        output = model_bart(
            input_ids=tokens_val["input_ids"][i].unsqueeze(0),
            attention_mask=tokens_val["attention_mask"][i].unsqueeze(0)
        )
        embedding = output.last_hidden_state.mean(dim=1).squeeze(0)
        embeddings_val.append(embedding)

# -----------------------------
# 7) Stack lists into big tensors for training
# -----------------------------
# X_*: inputs (embeddings); y_*: targets (0 or 1)
X_train = torch.stack(embeddings_train)                       # shape: [N_train, 768]
y_train = torch.tensor(df_train["spam"].values, dtype=torch.float32).unsqueeze(1)  # shape: [N_train, 1]

X_val = torch.stack(embeddings_val)                           # shape: [N_val, 768]
y_val = torch.tensor(df_val["spam"].values, dtype=torch.float32).unsqueeze(1)      # shape: [N_val, 1]

# -----------------------------
# 8) Build a tiny classifier (1 linear neuron)
# -----------------------------
# INPUT size = 768 (BART embedding size)
# OUTPUT size = 1  (a spam score; higher means "more spammy")
model = nn.Linear(X_train.size(1), 1)

# Loss function for yes/no problems on raw scores ("logits")
loss_fn = nn.BCEWithLogitsLoss()

# Optimizer = the "coach" that nudges weights to reduce loss
# lr (learning rate) = the step size for each nudge
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

# -----------------------------
# 9) Train: guess → measure error → learn a little (repeat)
# -----------------------------
for i in range(10000):
    model.train()                 # training mode (enables gradients)
    optimizer.zero_grad()         # reset old gradient info
    outputs = model(X_train)      # raw scores for each message
    loss = loss_fn(outputs, y_train)  # how wrong are we?
    loss.backward()               # compute how to change weights (backprop)
    optimizer.step()              # apply a tiny change to weights

    if i % 1000 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.4f}")

# -----------------------------
# 10) Evaluate with simple metrics students can read
# -----------------------------
def evaluate_model(X, y):
    """
    - Sigmoid turns raw scores into probabilities (0..1).
    - Threshold (here 0.25) turns probability into a yes/no prediction.
    - We print:
        * Accuracy    : overall % correct
        * Sensitivity : % of real spam we caught (a.k.a. recall for spam)
        * Specificity : % of real ham we kept as ham (recall for ham)
        * Precision   : when we said "spam", how often we were right
    """
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X))  # 0..1 spam probability
        y_pred = probs > 0.25            # pick a threshold (try 0.5 too)

        accuracy    = (y_pred == y).float().mean().item()
        sensitivity = (y_pred[y == 1] == y[y == 1]).float().mean().item()
        specificity = (y_pred[y == 0] == y[y == 0]).float().mean().item()
        precision   = (y_pred[y_pred == 1] == y[y_pred == 1]).float().mean().item()

        print(f"Accuracy   : {accuracy:.3f}")
        print(f"Sensitivity: {sensitivity:.3f}")
        print(f"Specificity: {specificity:.3f}")
        print(f"Precision  : {precision:.3f}")

print("Evaluating on the training data")
evaluate_model(X_train, y_train)

print("Evaluating on the validation data")
evaluate_model(X_val, y_val)

# -----------------------------
# 11) Try a few custom messages
# -----------------------------
custom_messages = pd.Series([
    "We have released a new product, do you want to buy it?",
    "Winner! Great deal, call us to get this product for free",
    "Tomorrow is my birthday, do you want to come to the party?"
])

print("Transforming custom messages...")
tokens_custom = tokenize_texts(custom_messages)

embeddings_custom = []
model_bart.eval()
with torch.no_grad():
    for i in tqdm(range(len(tokens_custom["input_ids"]))):
        output = model_bart(
            input_ids=tokens_custom["input_ids"][i].unsqueeze(0),
            attention_mask=tokens_custom["attention_mask"][i].unsqueeze(0)
        )
        embedding = output.last_hidden_state.mean(dim=1).squeeze(0)
        embeddings_custom.append(embedding)

X_custom = torch.stack(embeddings_custom)

# Predict spam probabilities for the custom messages
with torch.no_grad():
    model.eval()
    pred = torch.sigmoid(model(X_custom))
    print("Spam probabilities (0..1):")
    print(pred)
