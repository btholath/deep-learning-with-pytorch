from transformers import BartTokenizer, BartModel
import torch
from torch import nn

# -------------------------------
# 1) Example messages
# -------------------------------
messages = [
    "We have release a new product, do you want to buy it?", 
    "Winner! Great deal, call us to get this product for free",
    "Tomorrow is my birthday, do you come to the party?",
]

# -------------------------------
# 2) Tokenizer: words → numbers
# -------------------------------
# BART has its own tokenizer (like a dictionary).
# It breaks each message into tokens (subwords) and maps them to IDs.
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

out = tokenizer(
    messages,
    padding=True,       # make all sequences same length by adding "padding"
    max_length=512,     # cut off very long messages
    truncation=True,
    return_tensors="pt" # return PyTorch tensors (not lists)
)

# out now contains:
#   - input_ids: numeric tokens for each message
#   - attention_mask: 1s where there are words, 0s where there is padding
print("Tokenizer output (input_ids + attention_mask):")
print(out)

# -------------------------------
# 3) Load the pretrained BART model
# -------------------------------
# BART is a big Transformer trained on huge amounts of text.
bart_model = BartModel.from_pretrained("facebook/bart-base")

# -------------------------------
# 4) Get embeddings (mean pooling)
# -------------------------------
with torch.no_grad():   # we are not training, just using the model
    bart_model.eval()   # evaluation mode
    pred = bart_model(
        input_ids=out["input_ids"], 
        attention_mask=out["attention_mask"]
    )

    # pred.last_hidden_state has shape [batch_size, seq_len, hidden_size]
    # It’s a big 3D tensor: every word in every message has a vector
    # We average (mean) across the sequence to get one vector per message
    embeddings = pred.last_hidden_state.mean(dim=1)

    print("\nEmbeddings shape (batch_size, hidden_size):", embeddings.shape)
    print("\nVector for the 1st message:")
    print(embeddings[0, :5])  # print only first 5 numbers for readability


"""
What happens step by step
Messages: We give the model a few text messages.
Tokenizer: Converts text → numbers (input_ids). Each word (or subword) is assigned a unique ID.
BART model: A giant pretrained Transformer that understands context. It converts token IDs into embeddings (dense numeric vectors).
Mean pooling: We average word embeddings → get one vector per message (like a "summary number list").
Embeddings: Each message becomes a 768-dimensional vector (since bart-base has hidden size = 768).

Purpose
Why do this? Models like BART give us semantic embeddings.
- Messages that “mean the same thing” → embeddings are close.
- Spam vs. ham messages can be separated based on these vectors.

Real world:
- Spam detection
- Chatbots (understanding intent)
- Search engines (matching questions with answers)
- Recommendation systems (similar messages → similar suggestions)

"""
