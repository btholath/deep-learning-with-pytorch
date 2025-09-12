"""
BartModel.from_pretrained("facebook/bart-base") is the Hugging Face shortcut to download a ready-made BART model (with weights learned from massive text corpora) so you don’t have to train from scratch.
PURPOSE:
Show what BartModel.from_pretrained() does.

It downloads a pretrained BART Transformer (trained by Facebook AI on huge text data),
loads its weights, and makes it ready to use for embeddings or downstream tasks.

REAL-WORLD:
This is like reusing a trained "text brain" that already understands English,
so we don’t have to teach it language from zero.
"""

from transformers import BartTokenizer, BartModel
import torch

# -------------------------------
# 1) Load pretrained tokenizer + model
# -------------------------------
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# from_pretrained downloads pretrained weights & configuration
bart_model = BartModel.from_pretrained("facebook/bart-base")

# -------------------------------
# 2) Tokenize some text
# -------------------------------
text = "Hugging Face makes transformers easy to use."
inputs = tokenizer(text, return_tensors="pt")

print("Token IDs:", inputs["input_ids"])

# -------------------------------
# 3) Pass text through BART
# -------------------------------
with torch.no_grad():  # we are not training, just using the model
    outputs = bart_model(**inputs)

# outputs.last_hidden_state = embeddings for each token
print("Shape of last hidden state:", outputs.last_hidden_state.shape)

# -------------------------------
# 4) Compare with an untrained model (for demo)
# -------------------------------
# If we built a new BART without pretrained weights,
# it would start with random numbers, not useful embeddings.
bart_random = BartModel(bart_model.config)  # same config, but random weights
with torch.no_grad():
    outputs_random = bart_random(**inputs)

print("First 5 numbers from pretrained BART:", outputs.last_hidden_state[0, 0, :5])
print("First 5 numbers from random BART    :", outputs_random.last_hidden_state[0, 0, :5])

"""
What this demo shows

Pretrained model: BartModel.from_pretrained("facebook/bart-base")
    - Downloads weights trained on large datasets (CNN/DailyMail, books, Wikipedia, etc.).
    - Already “understands” English language patterns.

Random model: BartModel(bart_model.config)
    - Same architecture but random weights.
    - Produces meaningless embeddings until trained.

Why useful?
    - Instead of spending months training on billions of words, we can reuse pretrained knowledge.
    - Then fine-tune on specific tasks: spam detection, summarization, question answering, etc.

✅ When you run the script, you’ll see:
    - Pretrained embeddings are stable and non-random.
    - Random model outputs are just noise.
"""