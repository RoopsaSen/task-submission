The 4th order Runge-Kutta method has been used to iteratively solve the given set of equations. 
This method is one of the most accurate methods of iterative differential equation solving, giving accuracy upto 4th order in the Taylor series expansion
As the accuracy of the integrator depends on the timestep of the iteration(dt), with higher dt(0.1 or 0.01 seconds) the solution was tending towards infinity due to inaccuracies. With dt = 0.001 the solution converged. 
