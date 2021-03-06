# MLP_Parallel

This is the parallel implementation of an arbitrary MLP using back propagation algorithm. The parallel algorithm divides the training examples to different threads to parallelize the algorithm. The threads for different training examples communicate at the end of each iteration to update the parameters.

The algorithm is implemented using c and pthread for parallelization. The MLP implemented has one hidden layer with four neurons and one output layer, tanh and sigmoid functions used as activation function for the hidden layer and output layer respectively. Each thread gets equal amount of training example and they store their results on 2D array (each row represents a thread and column represents gradient descent for each parameter) and this array is used to calculate the total gradient descent for each parameter, this is how they communicate. Four threads are used here.

The algorithm is implemented to train an MLP to detect a cat in a picture. The MLP has one hidden layer with four neurons and one output layer. The inputs are pixel values of a 64*64 RGB picture (12288 input features) and there are 208 pictures for training.

To run the program compile it using gcc with -lpthread option

Acknowledgment

The read_csv function is adopted from manoharmukku we would like to thank him.
