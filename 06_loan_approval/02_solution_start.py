import sys
import torch
from torch import nn
import pandas as pd

df = pd.read_csv("/workspaces/deep-learning-with-pytorch/06_loan_approval/data/loan_data.csv")
df = df[["loan_status", "person_income", "loan_intent", "loan_percent_income", "credit_score"]]