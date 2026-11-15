## Pendulum - week 1
- This code numerically integrates the path of a particle through a potential. 
- The function RK4 uses the derivatives (calculated by hand here) of the particles position and velocity, given by the equations of motion, to predict the next position and velocity of the particle.
- The particles all start from x=-L with horizontal velocity of 2.5 and vertical velocity between -1 and 1.
- The RK4 function is applied repeatedly to create a trajectory of the particle from x=-L to x=L. These trajectories are shown on the figure below along with the potential.
<img src="Week1figure.png" width="50%"> 

## Pendulum - week 2
- This code integrates the pendulum equation of motion using the RK4 method from week 1.
- Calculates and compares the finite difference derivative and the derivative calculated using jax.grad. 
- To estimate l so that the pendulum reaches a final angle of theta_f, the newton method for root finding is used. Starting with an initial value of g/l, the predicted final angle is calculated and compared with theta_f. The finite-difference derivative is then used to update the value of g/l. This process is repeated for 10 iterations before converting the final estimate of g/l into a value of l, and returning the final angle and estimate for l.

## Pendulum - week 2/3
- A set of angle data points is created using a known value of l and RK4 formula on the equations of motion, with a small amount of random noise added for realistic effect.
- To estimate g/l from the measured data points the newton method for optimisation is used. Starting from an initial value of g/l, the error between the measured and simulated angles is calculated. The first and second derivatives of the error are then used to update the value of g/l. This process is repeated with the new value of g/l for 50 iterations before converting the final estimate for g/l into a value of l, which is then compared to the actual value of l.

## Neural network - week 4
- A set of data points is created by using a random generator to add Gaussian noise to the function evaluated values.
- The initial weights and biases are randomly generated rather than being set to zero. This is because if all the weights were zero, then they would all update the same, which cancels out the benefit of having 16 neurons compared to 1 neuron.
- The weights and biases are applied to the input value, with tanh being applied at each layer except the last one. The output layer does not use tanh because the final output does not need to be restricted to between -1 and 1.
- The loss function and its derivatives with respect to each weight and bias are then calculated. These derivatives are then used by the optimiser to update the weights and biases to reduce the loss function. This process is repeated for 3000 iterations before the final weights and biases are used to plot the predicted function, as shown on the figure below. The loss is recorded for each iteration and plotted on a logarithmic scale against iteration number, as shown below.
- Comment: Increasing the number of layers or neurons gives the network more parameters, which enables it to fit the training data more accurately and reduces the loss. However, if the network is too complex it can begin to learn the random noise in the training data as well as the underlying function. This is known as overfitting, and can mean that the network does not represent the true function as accurately.

<img src="Week4figure.png" width="45%"> <img src="Week4lossfigure.png" width="45%">


### Training on half the data points and test on the other half
- To investigate how well the neural network predicts points it has not been trained on, I split 100 data points into training and testing groups, each containing 50 data points.
- The neural network follows the true function closely, but misses some of the smaller details. For example, it does not always capture the extended peaks and troughts accurately. This is partly due to the placement of the training data around these points. Overall, since the network is able to make accurate predictions for the unseen data point, it suggests that the network has learnt the shape of the underlying function rather than just fitting the given data points.
- ADD FIGURE

### Decrease the number of training points
- The original training data used 50 data points. To investigate how reducing the number of training points affects the neural network, I changed this to 10 data points, keeping all other parameters the same.
- The neural network does not follow the true function as accurately as with 50 data points. The 10 data points are unable to represent every peak and trough of the true function, so the network has less information to learn about the oscillations between data points. However, the network does capture some of the oscillatory behaviour and models the overall shape well. 
- ADD FIGURE

### Increasing the range of x
- The original training data used 50 data points between -pi and pi. To investigate how well the neural network can fit the function over a wider range of x values, I changed this to 50 data points between -2pi and 2pi, keeping all other parameters the same.
- For larger values of x, the function oscillates more rapidly. This makes it more difficult for the neural network to fit the function accurately. This can be seen in the figure below, as the network remains relatively accurate for smaller values of x, but fails to capture the oscillations for larger x values.
- ADD FIGURE






