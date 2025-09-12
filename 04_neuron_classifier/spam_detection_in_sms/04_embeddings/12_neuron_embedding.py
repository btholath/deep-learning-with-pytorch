"""
GOAL:
Use a modern language model (BART) to turn SMS messages into number vectors
(= "embeddings"), then teach a simple neuron to tell SPAM (1) from HAM/not-spam (0).

WHY THIS MATTERS (real world):
- Email/SMS apps use models like this to filter spam.
- Search and chatbots use embeddings to understand meaning, not just exact words.
"""

import sys
import torch
from torch import nn
import pandas as pd
from transformers import BartTokenizer, BartModel
from tqdm import tqdm  # shows a progress bar while we loop
from sklearn.feature_extraction.text import CountVectorizer  # (not used below, but often used in earlier lessons)

# ---------------------------------------------------------
# 1) Load a pretrained "text brain": BART tokenizer + model
# ---------------------------------------------------------
# Tokenizer = breaks text into pieces (tokens) and maps them to numbers (IDs).
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Model = a large neural network that already learned English patterns from huge text datasets.
# from_pretrained downloads those learned weights so we don't start from zero.
bart_model = BartModel.from_pretrained("facebook/bart-base")

# ---------------------------------------------------------
# 2) Helper function: messages -> embeddings (meaningful numbers)
# ---------------------------------------------------------
def convert_to_embeddings(messages):
    """
    IDEA:
    - For each message, tokenize it (words -> IDs).
    - Run through BART to get token embeddings (vectors for each token).
    - Average across tokens to make ONE vector per message (mean pooling).
    RESULT:
    - A tensor of shape (num_messages, 768) because bart-base has 768-dim embeddings.
    """
    embeddings_list = []

    # tqdm shows a progress bar so students can see it's working
    for message in tqdm(messages):
        # Tokenize ONE message into tensors BART expects
        out = tokenizer([message],
                        padding=True,       # pad to the same length (safe for batches)
                        max_length=512,     # cut very long texts
                        truncation=True,
                        return_tensors="pt" # return PyTorch tensors
        )

        # We are NOT training BART here, just using it (faster, uses less memory)
        with torch.no_grad():
            bart_model.eval()  # switch to "evaluation" mode (no dropout etc.)

            # Pass token IDs through BART to get hidden states (embeddings per token)
            pred = bart_model(
                input_ids=out["input_ids"],
                attention_mask=out["attention_mask"]
            )

            # pred.last_hidden_state shape: [batch, seq_len, hidden_size]
            # Take the average over the token dimension (dim=1) to get ONE vector per message
            message_embedding = pred.last_hidden_state.mean(dim=1).reshape((-1))

            # Save this message's embedding
            embeddings_list.append(message_embedding)

    # Stack all message embeddings into one big matrix: [num_messages, 768]
    return torch.stack(embeddings_list)

# ---------------------------------------------------------
# 3) Load the SMS Spam dataset and make train/val splits
# ---------------------------------------------------------
# File has 2 columns per row (tab-separated):
#   - "ham" or "spam"
#   - the SMS text
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection",
                 sep="\t",
                 names=["type", "message"])

# Make a True/False label: True for spam, False for ham
df["spam"] = df["type"] == "spam"
df.drop("type", axis=1, inplace=True)  # we no longer need the text label column

# Split into:
#   - TRAIN (80%) to teach the neuron
#   - VAL (20%) to fairly test how well it learned
df_train = df.sample(frac=0.8, random_state=0)
df_val   = df.drop(index=df_train.index)

# ---------------------------------------------------------
# 4) Convert text messages to embeddings for train and val
# ---------------------------------------------------------
# NOTE: This step can take a bit because BART is big.
X_train = convert_to_embeddings(df_train["message"].tolist())
X_val   = convert_to_embeddings(df_val["message"].tolist())

# Make target tensors (1 = spam, 0 = ham) as a column vector
y_train = torch.tensor(df_train["spam"].values, dtype=torch.float32).reshape((-1, 1))
y_val   = torch.tensor(df_val["spam"].values,   dtype=torch.float32).reshape((-1, 1))

# ---------------------------------------------------------
# 5) Build a tiny classifier (just one neuron layer)
# ---------------------------------------------------------
# INPUT size  = 768 (embedding length)
# OUTPUT size = 1   (spam score; higher = more spammy)
model = nn.Linear(768, 1)

# Loss function for yes/no problems on raw scores ("logits")
loss_fn = torch.nn.BCEWithLogitsLoss()

# Optimizer (the "coach") that nudges weights to reduce the loss
# lr = learning rate = step size for each nudge
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

# ---------------------------------------------------------
# 6) Train the model: guess -> check error -> correct -> repeat
# ---------------------------------------------------------
for i in range(0, 10000):
    optimizer.zero_grad()          # clear old gradient info
    outputs = model(X_train)       # raw scores for each train message
    loss = loss_fn(outputs, y_train)# how wrong are we?
    loss.backward()                # compute how to adjust the weights (backpropagation)
    optimizer.step()               # move weights a tiny step to reduce loss

    if i % 1000 == 0:
        print("step", i, "loss", loss.item())

# ---------------------------------------------------------
# 7) Evaluation helper: turn scores -> probabilities -> yes/no, then report metrics
# ---------------------------------------------------------
def evaluate_model(X, y):
    """
    - Sigmoid turns scores into probabilities (0..1).
    - Threshold (0.25 here) turns probabilities into labels (spam/ham).
    - We print a few easy metrics:
        * accuracy   : overall percent correct
        * sensitivity: of all spam messages, how many we caught (recall for spam)
        * specificity: of all ham messages, how many we kept as ham (recall for ham)
        * precision  : of messages we called "spam", how many were truly spam
    """
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X))     # 0..1 probability of spam
        y_pred = probs > 0.25               # choose a threshold (try 0.5 too)

        print("accuracy   :", (y_pred == y).type(torch.float32).mean())
        print("sensitivity:", (y_pred[y == 1] == y[y == 1]).type(torch.float32).mean())
        print("specificity:", (y_pred[y == 0] == y[y == 0]).type(torch.float32).mean())
        print("precision  :", (y_pred[y_pred == 1] == y[y_pred == 1]).type(torch.float32).mean())

# ---------------------------------------------------------
# 8) See how we did on train and validation sets
# ---------------------------------------------------------
print("\nEvaluating on TRAIN data")
evaluate_model(X_train, y_train)

print("\nEvaluating on VAL data (new to the model)")
evaluate_model(X_val, y_val)

# ---------------------------------------------------------
# 9) Try a few custom messages to see real predictions
# ---------------------------------------------------------
X_custom = convert_to_embeddings([
    "We have release a new product, do you want to buy it?",
    "Winner! Great deal, call us to get this product for free",
    "Tomorrow is my birthday, do you come to the party?"
])

model.eval()
with torch.no_grad():
    custom_probs = torch.sigmoid(model(X_custom))  # probabilities of spam
    print("\nCustom message spam probabilities (0..1):")
    print(custom_probs)


"""
Embeddings are “meaning numbers.” Two messages with similar meaning will have similar embeddings even if they don’t share exact words.

Train/val split is about being fair: test on messages the model hasn’t seen.

Sigmoid + threshold turns a score into a probability, then into a yes/no decision.

Metrics (accuracy, sensitivity, specificity, precision) help us judge the model from different angles—important for real-world systems like spam filters.
"""