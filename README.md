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


# Pandas
