from numpy import *
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from time import sleep

learning_rate = 0.00001
num_iterations = 1000

lines = array([[0, 0]])

def compute_error(b, m, points):
    error = 0

    for l in points:
        x = l[0]
        y = l[1]
        error += (y - (m * x + b)) ** 2

    return error / float(len(points))

def step_gradient(b, m, points):
    b_gradient = m_gradient = 0
    N = float(len(points))

    for l in points:
        x = l[0]
        y = l[1]
        b_gradient += -(2/N) * (y - ((m * x) + b))
        m_gradient += -(2/N) * (x * (y - ((m * x) + b)))

    return [(b - (learning_rate * b_gradient)), (m - (learning_rate * m_gradient))];

def gradient_descent_runner(points):
    global lines
    b = m = 0

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points))
        lines = append(lines, [[b, m]], axis=0)

    return [b, m]

def run():
    # points = genfromtxt('brain_body.csv', delimiter=';', skip_header=1)
    points = genfromtxt('data.csv', delimiter=',')

    print "Starting gradient descent at error = {0}".format(compute_error(0, 0, points))
    print "Running..."
    [b, m] = gradient_descent_runner(points)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error(b, m, points))

    xarr = points[:,0]
    yarr = points[:,1]

    fig1 = plt.figure()
    plt.scatter(xarr, yarr)
    l, = plt.plot([], [], 'r-')
    plt.xlim(-10, max(xarr)+10)
    plt.ylim(-10, max(yarr)+10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression - Gradient Descent')
    line_ani = animation.FuncAnimation(fig1, update_line, len(lines), fargs=(lines, xarr, l), interval=300, blit=True)

    plt.show()
    return

def update_line(num, lines, xarr, line):
    b, m = lines[num]

    res = lambda e: m*e + b
    new_y = array([res(x) for x in xarr])

    line.set_data(xarr, new_y)
    return line,

if __name__ == '__main__':
    run()
