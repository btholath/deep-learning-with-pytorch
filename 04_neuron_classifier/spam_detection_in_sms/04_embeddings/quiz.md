# Spam Detection Techniques

 key techniques like vectorization, model validation, and performance evaluation

 Why is it important to convert a sparse matrix to a dense format before using it in PyTorch models?

PyTorch requires dense formats for certain operations that sparse formate cannot support.

What is the significance of the max_features parameter in the Count Vectorizer tool?

It limits the number of most frequent words considered
"max_features" restricts the vocabulary size to the top N most frequent words, helping to focus model training on the most relevant data.

How does the sigmoid activation function affect the output of a spam detection model?
It maps the model's output to a probability scale betweeb 0 and 1, indicating the likehood of spam


What is the purpose of using validation data in machine learning models like spam detection?
To tune model hyperparameters and prevent overfitting, ensuring the model performs well on unseen data.

What role does model evaluation play at the end of a spam detection project ?
To verify the model's performance on unseen data and ensure it generalizes well.
Model evaluation test the trained model against new, unseen data to assess its effectiveness outside of the training environment.

Which metrics are commonly used to evaluate a spam detection model?
Accuracy, sensitivity, specficity and precision
These metrics provide a comprehensive assessment of model performance, measuring correct classifications, and the ability to
detect positive and negative instances correctly.





