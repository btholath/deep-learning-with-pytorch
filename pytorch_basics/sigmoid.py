import torch
import torch.nn.functional as F

# Example raw outputs from a model (logits)
raw_outputs = torch.tensor([[-3.5], [0.2], [4.7]])

# Apply sigmoid to turn them into probabilities
probabilities = F.sigmoid(raw_outputs)

# Print results
print("Raw outputs (logits):")
print(raw_outputs)

print("\nAfter applying sigmoid (probabilities):")
print(probabilities)

# Extra: show interpretation
for raw, prob in zip(raw_outputs, probabilities):
    print(f"Raw: {raw.item():.2f} → Sigmoid: {prob.item():.2f} "
          f"({'spam' if prob.item() > 0.5 else 'ham'})")


"""
The raw outputs can be any number (negative, positive, or zero).

The sigmoid squishes those numbers into a range between 0 and 1.

Now we can read them as probabilities:

0.03 → almost certainly ham
0.55 → more likely spam
0.99 → almost certainly spam


Imagine the model gives each message a “spam score.”
But the score could be anything (like -10, 0.2, 12).
The sigmoid is like a squishy S-shaped slide that pushes every score into a safe range between 0 and 1.

Now we can say:
0.03 → “3% chance spam”
0.55 → “55% chance spam”
0.99 → “99% chance spam”

Why do we do this?
Without sigmoid: raw scores are hard to interpret (like -3.5 or 4.7).
With sigmoid: we get clean probabilities:
Near 0 → ham (not spam)
Near 1 → spam

Then we can set a threshold (commonly 0.5):
If probability > 0.5 → classify as spam
Else → classify as ham

"""