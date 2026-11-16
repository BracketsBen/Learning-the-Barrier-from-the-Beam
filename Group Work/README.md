# The inverse problem
## Task 1:
It would take a particle $\frac{10}{3}$ $\approx  3.3$    seconds to cross <br>
We would need T to be more because the potential would likely slow it down. 

## Task 2:


## Task 3:
Both plots produced by the code behave as expected, as well as with different potential inputs. 
Yet to code the energy to prove constant


## Task 5:
vy0 is varied in addition to y0 to provide more information about the shape of the potential. Since the particles start with different positions and directions, they interact with different parts of the potential. This means that the potential reconstructed by the network should be a more accurate representation of the true potential. 

## Task 6:
Dividing (x,y) by L scales the coordinates to approximately the range [-1,1]. The tanh function used in the layers is more responsive to small input values, meaning that small changes in the input produce a larger difference in its output. So by scaling the coordinates, the network will learn the potential more efficiently.

The softplus function returns the value log(1+e^x) for any real input x. This function is strictly positive, which therefore guarantees that V is always positive. This is necessary since otherwise the network could return negative values for V, which is not wanted in this problem.

At the start of training, the network uses its initial weights and biases to produce a raw output. This raw output is expected to be around 0, since the weights are centered around 0 and all the biases are initially 0. Subtracting 2 from this raw value shifts the softplus function right, resulting in a smaller initial potential. Overall, this provides the network with a small, positive initial value for the potential, which can then be adjusted during training to better match the true potential.

## Task 8:
Dividing by the noise squared shows how large the errors are compared with the expected noise in the observed values. Therefore, an error is only significant if it is large relative to the expected noise.

A loss of 1 would mean that the network fits the data as well as the noise allows. This is because the errors would be approximately the same size as the expected noise in the observed values.
