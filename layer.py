# Abstract layer class 
class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None 
    
    def feed_forward(self, input):
        """
        Computes outputs Y for given inputs X in forward propagation.
        X -> Layer -> Y
        """
        raise NotImplementedError

    def back_propagation(self, output_error, learning_rate):
        """
        Computes dE/dX for a give dE/dY and updates the learnable parameters.
        """
        raise NotImplementedError