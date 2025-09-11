"""
The z-score normalization formula is one of the most common ways we scale data so it’s easier for machine learning models to work with.

For a value 𝑥

    z = ( x−μ​ ) / σ 

Where:
    𝑥 = the original value
    μ (mu) = the mean (average) of the dataset
    σ (sigma) = the standard deviation of the dataset


What it does
    Subtracting the mean ( 𝑥 − 𝜇 )  → recenters the data around 0
    Dividing by standard deviation (𝜎) → scales it so most values fall between -3 and +3

So:
    Values greater than the mean → positive z-scores
    Values less than the mean → negative z-scores
    A z-score tells you how many standard deviations away from the average a value is    
    
"""
import torch

prices = torch.tensor([15000., 20000., 25000., 30000.])

mean = prices.mean()        # average = 22500
std = prices.std()          # standard deviation σ

z_scores = (prices - mean) / std
print(z_scores)

"""
Z-score normalization answers: “How unusual is this number compared to the group?”
0 = exactly average
Positive = above average
Negative = below average
"""

"""
Given the code below, what steps are needed to perform Z-score normalization on price data stored in variable price?
    price_mean = price.mean()
    price_std = price.std()
    price = (price - price_mean) / price_std

Substract the mean and divide by the standard deviation
Z-score normalization centers the data by structuring the mean and scaling it by dividing by the standard deviation.
This step-by-step transformation standarizes the values to a range that stabilizes learning.
"""