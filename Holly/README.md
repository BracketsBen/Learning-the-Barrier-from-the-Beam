## Pendulum week 1
- Numerically integrates the path of a particle through a potential. 
- The function RK4 uses the derivatives (calculated by hand in this case) to predict the next position and velocity of the particle.
- The particles all start from x=-L with horizontal velocity of 2.5 and vertical velocity between -1 and 1.
- The RK4 step is then looped to create a trajectory of the particle from x=-L to x=L.
![Particle trajectories through a potential](Week1figure.png)

---

## Pendulum week 2
- Integrates the pendulum equation of motion using the RK4 method from week 1.
- Calculates and compares the finite difference derivative and the grad derivative.
- To estimate l so that the pendulum reaches a final angle of theta_f, use the newton method where the function, f, is the difference between the predicted final angle and theta_f. Starting with an initial value of l, calculate the ratio g/l and the predicted final angle. Then use the derivative of f to update g/l. Loop over this process with the new value of g/l.

