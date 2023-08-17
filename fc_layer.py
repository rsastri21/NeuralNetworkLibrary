from layer import Layer 
import numpy as np

# Inherit from base Layer class
class FCLayer(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        input_size: int containing the number of input neurons.
        output_size: int containing the number of output neurons.
        
        Maintains weights and bias as fields. Weights and bias are
        initially randomly spread between -0.5 and 0.5.

        Dimensionality:
            Weights matrix is of size j x k with j input neurons and k output neurons.
            Bias is a column vector of size 1 x k with k output neurons.
        """
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
    
    def feed_forward(self, input_data):
        """
        Returns the output from z = aW + b with a corresponding to the provided input data.
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output
    
    def back_propagation(self, output_error, learning_rate):
        """
        Computes the appropriate derivatives and updates parameters before 
        returning the error with respect to the input.
        """
        # Return value
        input_error = np.dot(output_error, self.weights.T)
        # Value to update weights
        weights_error = np.dot(self.input.T, output_error)

        # Update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error # output_error is equivalent to dE/db

        return input_error