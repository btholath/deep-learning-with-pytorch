
Train / Validation / Test Data
Evaluate a model - how good is the model ? Should we change something in the structure (hyperparameters) ?

- Training data (60-70%)
    - Already used for training the model
    - We can't use this data for validating the performance
- Validation data (10-20%)
    - Used for selecting the model architecture and decining when to stop training
    - Used to tune the "hyperparameters"
    - This data has already been "seen" by our algorithm, once we chose a mode.
- Test data (10-20%)
    - We need completely new data to answer this question
    - We could've held back data from the start to use for test data
    - Or we can collect new data now
    