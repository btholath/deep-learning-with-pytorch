1. Single neuron first → students grasp weights, bias, logits, sigmoid, loss.
2. Sequential small nets → concrete upgrade: hidden layers & activations.
3. Swap activations → see how one line changes training dynamics.
4. Optimizers (Adam) → speed & stability improvements feel real.
5. Mini-batches → how pros train on real datasets.
6. Epoch tracking → good habits for reading training curves.
7. Train/Val → scientific thinking: fair testing.
8. Manual two-layer → demystify what Sequential automates.

 understanding of neural networks, emphasizing the structural and learning mechanisms that underpin their functionality. Explore how neurons are layered to form networks, the role of backpropagation in weight adjustment, and why this process is vital for training.

 What is the primary goal of stacking multiple neurons in a neural network?
 To learn more complex relationships
 Stacking neurons allows the network to capture and model more complex patterns and interactions in the data, which a single neuron cannot do.

 How are the weights adjusted in backpropagation?
 By computing the error from the output and propagating it backward
 Backpropagaion calculates the gradient of the loss function with respect to each weight by chaning derivates backward through the network, adjusting weights to miniimize loss.

 Why is backpropagation crucial for neural networks?
 It allows the network to optimize weight adjustments for improved accuracy
 Backpropagation effetively helps in minimizing the loss function, leading to better model performance and accuracy over training iterations.
 