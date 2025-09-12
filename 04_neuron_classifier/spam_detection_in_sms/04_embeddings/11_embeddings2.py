"""
GOAL:
Turn text messages into "embeddings" (number vectors)
using a pretrained Transformer model called BART.

BIG IDEA:
- Computers don’t understand words directly.
- BART has already learned English from millions of documents.
- We feed it our messages, and it gives us embeddings
  (big lists of numbers that capture meaning).
- Later, we can use these embeddings for tasks like spam detection,
  recommendation, or grouping similar messages.
"""

from transformers import BartTokenizer, BartModel
import torch
from torch import nn
from tqdm import tqdm   # shows a progress bar when looping

# -------------------------------
# 1) Some example messages
# -------------------------------
messages = [
    "We have release a new product, do you want to buy it?", 
    "Winner! Great deal, call us to get this product for free",
    "Tomorrow is my birthday, do you come to the party?",
]

# -------------------------------
# 2) Load pretrained BART
# -------------------------------
# The tokenizer breaks text into tokens (subwords) and maps them to IDs (numbers).
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# The model is a giant "text brain" trained on huge datasets.
# from_pretrained loads the already-learned weights.
bart_model = BartModel.from_pretrained("facebook/bart-base")

# -------------------------------
# 3) Function: convert messages → embeddings
# -------------------------------
def convert_to_embeddings(messages):
    embeddings_list = []   # will hold embeddings for each message
    
    # Loop over each message (tqdm shows a progress bar)
    for message in tqdm(messages):
        # Tokenize: turn message into input_ids (numbers) and attention_mask
        out = tokenizer(
            [message],
            padding=True,        # pad to same length
            max_length=512,      # cut off if too long
            truncation=True,
            return_tensors="pt"  # return PyTorch tensors
        )

        with torch.no_grad():   # no training, just inference (saves memory)
            bart_model.eval()   # set model to evaluation mode

            # Run the tokens through the model
            pred = bart_model(
                input_ids=out["input_ids"], 
                attention_mask=out["attention_mask"]
            )

            # pred.last_hidden_state has embeddings for every token (word piece)
            # We take the mean (average) over tokens to get 1 embedding per message
            embeddings = pred.last_hidden_state.mean(dim=1).reshape((-1))
            
            # Add this message's embedding to our list
            embeddings_list.append(embeddings)
    
    # Stack all embeddings into one tensor: shape = (num_messages, embedding_size)
    return torch.stack(embeddings_list)

# -------------------------------
# 4) Convert our messages
# -------------------------------
X = convert_to_embeddings(messages)

# Print the shape: (3 messages, 768 features per embedding)
print("Embeddings shape:", X.shape)


"""
Tokenizer = dictionary that changes words into numbers.

BART model = a pretrained Transformer that already understands English.

Embeddings = long number lists (768 numbers per message here) that represent the meaning of each message.

Mean pooling = we average word embeddings → one vector per full message.

Why useful?

Messages that mean similar things → embeddings will be close.

Helps in spam detection, chatbot understanding, recommendation systems, etc.
"""