#reference: https://www.youtube.com/watch?v=5CXhHx56COo
#running the following code will plot the position of the particle(governed by the equations given) in the last 1 second(1000 0.001 second timesteps)
#press escape to exit simulation

#importing libraries
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

#defining given constants
a = 10
b = 28
c = 2.667

#defining given change function on vector w = [x, y, z] 
def f(w):
    w_ = np.array([0.0, 0.0, 0.0])
    w_[0] = a*(w[1] - b)
    w_[1] = b*w[0] - w[1] - w[0]*w[2]
    w_[2] = w[0]*w[1] - c*w[2] 
    return w_

#finding [x, y, z] at time t+1 given [x, y, z] at time t
#fourth order Runge-Kutta method has been used to solve this system of ODE
def next_step(f, dt, w):
    f1 = f(w)
    f2 = f(w + (dt/2)*f1)
    f3 = f(w + (dt/2)*f2)
    f4 = f(w + dt*f3)
    w_ = w + dt*(f1 + 2*f2 + 2*f3 + f4)/6
    return w_

#defining [x, y, z] at time t = 0
w = np.array([0, 1, 1.05])
#dt is the time step after which the vector is evaluated
dt = 0.001

#initially t = 0
t = 0

#initialize arrays for storing path points
xdata = []
ydata = []
zdata = []
max_len = 1000

fig = plt.figure()
ax = plt.axes(projection='3d')

#running loop until escape key is pressed
while t >= 0:
    #add current positions to possition arrays
    xdata.append(w[0])
    ydata.append(w[1])
    zdata.append(w[2])

    #keep only upto 1000 elements in aaray(last second)
    if len(xdata) > max_len:
        xdata.pop(0) 
        ydata.pop(0)
        zdata.pop(0)

    #print(w)

    #find the position after timestep(dt) of 0.001 second
    w = next_step(f, dt, w)

    #increment time by dt
    t = t + dt

    #plot position points
    plt.cla()
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    ax.plot3D(xdata, ydata, zdata)

    plt.grid(True)
    plt.pause(0.001)

#show plot
plt.grid(True)
plt.show()