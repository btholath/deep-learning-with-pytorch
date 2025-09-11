# deep-learning-with-pytorch
Master PyTorch—build, train &amp; deploy models with real projects, Gradio apps &amp; ResNet-powered transfer learning.


# VS Code in GitHub's Codespace
# To execute python within Jupyter notebook.
# Step 1: Make sure Jupyter is installed
Open a terminal inside your Codespace and run:

@btholath ➜ /workspaces/deep-learning-with-pytorch (main) $ source .venv/bin/activate
@btholath ➜ /workspaces/deep-learning-with-pytorch (main) $ pip install --upgrade pip
@btholath ➜ /workspaces/deep-learning-with-pytorch (main) $ pip install notebook jupyter ipykernel

This ensures Jupyter and the kernel support is available inside your environment.

# Step 2: Add your Python environment as a kernel
Still in the terminal, run:
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"

The --display-name is what will show up in the kernel list in VS Code.


# Step 3: Restart VS Code / Reload Window
Go back to your notebook tab in VS Code.
At the top right, click “Select Kernel” (or it may already be prompting you).
Now you should see “Python (myenv)” or similar.
Select it, and then Shift+Enter should run your cells.
