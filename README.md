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

