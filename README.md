# deep-learning-with-pytorch
Master PyTorch—build, train &amp; deploy models with real projects, Gradio apps &amp; ResNet-powered transfer learning.


# VS Code in GitHub's Codespace
# To execute python within Jupyter notebook.
# Step 1: Make sure Jupyter is installed
Open a terminal inside your Codespace and run:

@btholath ➜ /workspaces/deep-learning-with-pytorch (main) $ # create & activate if you don't have one yet
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install jupyter ipykernel

# register with a friendly name
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"



2) Clear stale server state and token issues
Sometimes the runtime files get weird. Clear them:
rm -rf ~/.local/share/jupyter/runtime/*


Then restart:
jupyter lab --ip=0.0.0.0 --port 8888 --no-browser


Fix common version conflicts (fast safe reinstall)

A frequent cause of 500s is mismatched jupyter_server, notebook, nbclassic, or traitlets. Reinstall the core stack cleanly:

python -m pip install --upgrade pip
pip install --upgrade jupyterlab jupyter_server notebook nbclassic nbconvert nbformat traitlets tornado jinja2 markupsafe

Then relaunch (step 0).
Tip: If you only use Classic Notebook UI, ensure nbclassic is installed. If you use Lab, ensure jupyterlab + jupyter_server are consistent.




# Phase 1 – Explore & Represent Text

explore.ipynb → Explore the SMS dataset, count spam vs ham, peek at examples.
Idea: “Let’s see what spam looks like!”

cv.py neuron1  → Bag-of-Words with CountVectorizer (word counts → numbers).
Idea: show how text becomes a table of numbers.

tf-idf.py tf-idf  → TF-IDF representation.
Idea: explain “important words get bigger weights” (like “prize” vs “the”).


# Phase 2 – First Neurons & Sigmoid

sigmoid.ipynb → Simple sigmoid function demo.
Idea: “How do we turn any number into a probability between 0 and 1?”

neuron1.py

neuron1

 → First linear neuron trained on Bag-of-Words.
Idea: build the simplest classifier, see it output values before sigmoid.

neuron_sigmoid.py → Show outputs run through sigmoid.
Idea: predictions become probabilities for spam vs ham.



# Phase 3 – Training, Testing, Evaluating

neuron_evaluate.py → Train on the whole dataset and evaluate (overfitting).
Idea: show why evaluating only on training data is misleading.

neuron_train_val.py → Proper train/validation split with metrics.
Idea: introduce accuracy, precision, recall, specificity.

neuron_test.py → Try predictions on new custom SMS messages.
Idea: “Can our robot detect spam in real life?”


# Phase 4 – Embeddings (Dense Meaning Vectors)

embeddings.py → Extract embeddings from a pretrained model (BART).
Idea: show how text turns into a dense vector that captures meaning.

embeddings2.py → Wrap embedding extraction into helper functions.
Idea: organize embeddings neatly for many messages.

neuron_embedding.py → Train a linear classifier using embeddings.
Idea: compare performance with Bag-of-Words.

_neuron_bart.py → Full pipeline with embeddings + evaluation.
Idea: put it all together.




# Phase 5 – Transformers
Script: transformer.py
Use BERT tokenizer + embeddings.
Idea: “Here’s the modern upgrade—transformers!” Compare to earlier methods.




Optimizing Training for Our Neuron Classifier
When evaluating the neuron model, experimenting with reducing training iterations and increasing the learning rate can help reduce the time required to run the code. For example, training iterations can be reduced to 25,000, and the learning rate can be increased to 0.05.

To measure the time taken for training, the time library can be used. Start by importing it with the command:

import time
Then, create a timestamp at the beginning of the code:

t_start = time.time()
and another timestamp at the end:

t_end = time.time()
Finally, include the following command to calculate and display the total time:

print(f'It takes: {t_end - t_start}s for this program to finish.')
This will allow you to see how your chosen hyperparameters impact the training duration.


What is the primary purpose of converting message data into tensors before training the model?
To mae the data compatible with PyTorch for model training
Tensors are the primary data structure used in PyTorch, and converting data into this format ensures it can be effectively processed  and manipulated during model training.


What is the role of the sigmoid activation function in a neuron model?
To map the neuron's linear output to a range between 0 and 1, suitable for represneting probabilities.
The sigmoid function compresses outputs to the 0-1 interval, which is crucial for tasks that interpret these outputs as probabilities, such as in binary classification.

Referring to the provided graph of the sigmoid function, how does it respond to extreme input values?
It approaches 0 for negative inputs and 1 for positive inputs, effectively squashing extreme to the edges of the 0-1 range.
As seen in the graph, the sigmoid function's S-shaped curve shows that as the input values becomes extremely negative or positive, the functions's output asymptotically approaches the limit of 0 and 1 respectively. This characteristic ensures that all outputs are bounded within the 0 to 1 range, making it suitable for probablitity models.

# Loss Functions and Evaluation Metrics in Spam Detection
Implementation and impact of loss functions and metrics in spam detection models. Dive into why Mean Squared Error (MSE) is unsuitable for models using sigmoid activation, explore the advantages of Binary Cross Entropy (BCE) with logits loss, and learn how sensitivity is accurately calculated in spam detection.

Why is mean squared error (MSE) not suitable for training a binary classification model using a sigmoid activation function?
MSE can result in very small gradients when the predicted output is close to the target class, making learning inefficeint.
MSE is not ideal because as the predicted probability nears the acual class 0 or 1, the gradient of the loss with respect to the prediction becomes smaller.
This can slow down or even halt learning due to insufficeint updates to the weights.

What is the primary advantage of using Binary Cross Entropy (BCE) with logits loss over MSE in a model using sigmoid activation?
BCE with logits integrates the sigmoid activation directly into the loss calculation, which helps in managing issues with vanishing gradients.

How is sensitivity (true positive rate) calculated in a spam detection model?
Sensitivity measures how well the spam detection model identifies actual spam messages. It is specifically calculated by dividing the number or spam messages correctly
identified by the model *true positives) by the total number of atual spam messages, which includes both true positives and false negatives.

# Data Segmentation in Model Development
Understanding of the critical distinctions and purposes behind dividing data into training, validation, and test sets in artificial intelligence model development. You'll test your knowledge on why it’s essential to separate these datasets to prevent overfitting, ensure unbiased evaluation, and maintain the integrity of the model tuning process. This quiz will help solidify your grasp of how structured data management contributes to building robust, generalizable AI models.

What is the purpose of dividing data into training, validation and test sets in model development?
To ensure that the model is trained, validated and tested on different segments of data to avoid overfitting and to evaluate the moel's performance on unseen data.
The training set is used to fir the model, the valiation set is used to fine-tune the model parameters and select the best model architecture, and the test set is used to provide an unbiased evaluation of a final model fit on the training dataset.


Why is it important to keep the validation data separate from the training data?
To prevent data leakaga and ensure that the model tuning is based on unbiase data, which helps in genuinely evaluating the model's ability to generalize
Keeoing the validation data separate prevents the model from seeing this data during training, thereby providing a reliale way to check for overfitting and to tune the model's hyperparameters without bias.

What is the role of test data in model evaluation?
To assess the model's final performance on completely unseen data, providing an unbiased evaluation of its generalization ability.
Test data is used after the model has been trained and validated to measure how well it can perform on new, unseen data, reflecting its potential real-world performance.


# Enhancing Detection with LLM Embeddings
applying large language model (LLM) embeddings to boost spam detection accuracy. It covers the generation of embeddings with the BART model, the functionality of embedding conversion and the impact of LLM embeddings on spam filtering models. Dive into how these advanced techniques enhance the model's ability to interpret and classify text, providing a robust defense against evolving spam tactics.

What is the advantage of using embeddings from a large language model (LLM) for spam detection?
Embeddings capture the semantic meaning of texts, improving the model's ability to detect spam in varied and novel message forms.
By capturing deeper semantic meanings, embeddings allow the spam filter to recognize spam characteristics beyond basic keyword matching, adapting to new and unseen message types.

How are embeddings generated using the BART model in spam detection?
The BART model tokenizes the text and uses its layers to produce embeddings that capture contextual information from the input
The BART model processes tokenized text through multiple layers, generating embeddinhs that represemt the text in a meaniful, contextually enriched numerical format,

What is the purpose of the convert_to_embeddings() function in the context of LLMs?
To transform text messages into numerical embeddings using a pre-trained BART model, facilitating more effective spam detection
This function streamlines the process of converting text to embeddings, making it easier to integrate advanaced language model outputs into spam detection systems.

How does integrating embeddings from an LLM improve spam detection models?
Embeddings provide a deep semantic understaning of text, allowing the model to detect spam based on the underlying meaning rather than just surface features
By utilizing embeddings, spam detection models can more effectively identify spam messages that may not fit traditional patterns, increasing both accuracy and adaptability.

# Neuron Classifier
What does this code snippet demonstrate in the context of spam detection?

from sklearn.feature_extraction.text import CountVectorizer
 
# Sample texts
texts = ["Free money now!!!", "Hi, how are you today?"]
 
# Initialize CountVectorizer
vectorizer = CountVectorizer()
 
# Fit and transform the texts
feature_matrix = vectorizer.fit_transform(texts)
 
print(feature_matrix)

Conversion of text into a numeric token count matrix
The CountVectorizer converts text data into a matrix of token counts, which is crucial for training machine learning models to recognize patterns in text, such as potential spam indicators.

What is the purpose of dividing data into training and validation sets in spam detection?
To train the model on one set and tune hyperparameters on another to assess the model's performance on unseen data
The training set is used to fit the model, while the validation set is crucial for checking how well the model performs on new, unseen data. This step helps ensure that the model can generalize beyond the examples it has been trained on, providing an early indication of how it might perform under real-world conditions.


Please fill the blank field(s) in the statement with the right words.
The effectiveness of a spam detection model in correctly identifying non-spam messages is specifically measured by __.
Options: precision, recall, specificity, accuracy
Explanation
Specificity measures the true negative rate, which is critical for ensuring that legitimate messages are not incorrectly marked as spam.

What does this code primarily illustrate?
import torch
import torch.nn as nn
import torch.nn.functional as F
 
# Sample custom messages transformed into feature vectors
custom_messages = cv.transform([
    "Exclusive offer just for you, claim now!",
    "Urgent update required for your account safety"
])
 
X_custom = torch.tensor(custom_messages.todense(), dtype=torch.float32)
 
model.eval()
 
with torch.no_grad():
    pred = nn.functional.sigmoid(model(X_custom))
    print(pred)

Making a prediction using a trained model
Explanation
The code demonstrates how a trained spam detection model is used to predict the likelihood of spam for new, unseen messages, showcasing model application in a real-world scenario.

Which metrics are essential for effectively evaluating a spam detection model?
Sensitivity and Specificity
Sensitivity (or recall) measures the model's ability to identify all relevant instances (all actual spams), and specificity assesses how well the model identifies non-spam, both essential for balanced performance in spam detection.

Accuracy and Precision
Accuracy measures the overall correctness of the model across both classes, while precision assesses the correctness of positive predictions (spam), crucial for ensuring the reliability of spam detection.


Why is it important to shuffle the data before splitting into training and validation sets?
To ensure that the training and validation sets are representative of the overall dataset
Shuffling the data prior to splitting helps prevent any bias that might be introduced by the order of the data.

What characteristic of the sigmoid function is crucial for binary classification models like spam detectors?
Its capability to squash outputs to a range between 0 and 1
This is the key feature of the sigmoid function in binary classification. It compresses the output of the neuron to the range [0, 1], making it interpretable as a probability, which is essential for deciding between two classes (e.g. spam or not spam).


Please fill the blank field(s) in the statement with the right words.
To avoid bias towards any particular class, it is crucial to ensure that the dataset is balanced before __ a spam detection model.
Options: training, preprocessing, evaluating, deploying

Ensuring the dataset is balanced before training helps prevent the model from being biased towards the majority class. This is critical in spam detection where imbalance can significantly skew predictions, potentially leading to either a high number of false positives or false negatives. Balancing the dataset facilitates fair and accurate model training, improving overall performance.


What is the key reason for using a loss function like binary cross-entropy in spam detection models?
It effectively handles output probabilities between 0 and 1, suitable for binary outcomes
Binary cross-entropy is optimal for models outputting probabilities, as it measures the performance of a classification model whose output is a probability value between 0 and 1.



What does the sigmoid function compute in this context?

import torch
import torch.nn.functional as F
tensor = torch.tensor([1.0, -1.0, 0.5])
output = F.sigmoid(tensor)
This is how the sigmoid function looks like:
Conversion of tensor elements to values between 0 and 1
The sigmoid function applies an S-shaped logistic function to each element in the tensor, which maps any real-valued number into the range of 0 to 1. This property is particularly useful in binary classification tasks, such as spam detection, where outputs are interpreted as probabilities.


What are the key steps in preprocessing text data for spam detection?
Tokenization of sentences
Tokenization is a fundamental step in text preprocessing which involves splitting text into individual words or phrases, making it crucial for preparing input data for neuron processing.
Applying CountVectorizer
CountVectorizer converts text documents to a matrix of token counts, which is very useful for converting raw text into a format that can be fed into deep learning models, particularly if not using embedding layers directly.


What metric measures the proportion of actual positives that are correctly identified as such (true positive) in spam detection?
Sensitivity
Sensitivity measures how many actual positives are correctly identified, essential for ensuring no spam is missed.


What is the primary purpose of the activation function in a neuron model for spam detection?
To enable the model to make binary decisions
Activation functions like sigmoid are crucial in binary classification tasks as they map the output to a probability between 0 and 1, enabling binary decisions.


Please fill the blank field(s) in the statement with the right words.
The __ activation function is crucial for mapping outputs of the neuron to a probability between 0 and 1, suitable for binary classifications like spam detection.
Options: sigmoid, relu, tanh, softmax

sigmoid
The sigmoid function is essential in binary classification because it maps any input value to a probability between 0 and 1, ideal for distinguishing between two classes. This feature allows outputs to be directly interpreted as probabilities, facilitating clear decision-making based on a threshold, typically 0.5, to classify inputs as one class or another.


# Single Neuron Regression
What impact does removing outliers have on a regression model predicting used car prices?
It can help improve the model’s generalization by focusing on more typical data points
Removing outliers may help the model perform better on typical data by reducing the influence of anomalies.

Which feature is likely to have a higher influence on the price prediction in a used car dataset?
Mileage of the car
Mileage directly relates to the car's condition and usage, significantly impacting its market value.


In the context of used car prices, what would be a likely target variable for the model?
Note: The target variable is the variable that the model is designed to predict.
The price of the car
The price is the outcome variable that the model aims to predict.

Given the plot showing mileage on the x-axis and price on the y-axis, what trend does this plot suggest about the relationship between mileage and used car prices?
Prices decrease as mileage increases
This is a common trend where higher mileage often correlates with lower prices due to increased wear and decreased desirability.


Given the following PyTorch code:

import torch
# Sample input data
age = torch.tensor([10, 20, 30], dtype=torch.float32)
mileage = torch.tensor([5000, 15000, 25000], dtype=torch.float32)
 
# Combine age and mileage into a tensor
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(mileage, dtype=torch.float32)
])
 
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
What does the final tensor X contain?

A tensor with normalized values based on the mean and standard deviation of each column
The tensor X is updated such that each value is adjusted relative to the mean and scaled by the standard deviation of its respective column.


Please fill the blank field(s) in the statement with the right words.
For robust model training, ensuring that data is balanced or uniformly scaled across all input __ is vital to prevent bias in the deep learning algorithm.
Options: labels, segments, features, nodes
features
Uniform scaling or balancing prevents any one feature from dominating the model's decision process due to scale differences.


What metrics are suitable for evaluating a used car price prediction model?
Note: The target variable for prediction in this model is numerical (price).
Root Mean Squared Error (RMSE)
RMSE provides a clear indication of the average error magnitude, making it suitable for regression problems.

Mean Absolute Error (MAE)
MAE is another important regression metric that measures the average magnitude of errors in predictions, without considering their direction.


What does the term model normalization refer to in the context of predicting used car prices?
Scaling input features to a similar range
Normalization makes training more effective by ensuring that all features contribute proportionately.

What factors should be considered when selecting features for a model predicting used car prices?
Car mileage
Mileage is a critical determinant of a car's value, as it directly relates to wear and usage.
Number of previous owners
The number of previous owners can affect a car's market value, as fewer owners may suggest better maintenance and reliability.

What is the primary objective of using a single neuron model in predicting used car prices?
To predict prices based on vehicle age and miles driven
The model's goal is to use regression to predict prices from features like age and mileage.


Please fill the blank field(s) in the statement with the right words.
Effective handling of __ and categorical data is crucial for training accurate machine learning models, especially when predicting values like used car prices.
Options: textual, numerical, visual, audio
numerical
Proper management and processing of both numerical and categorical data ensure that the models are based on comprehensive and correctly formatted inputs.



