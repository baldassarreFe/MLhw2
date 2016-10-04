from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy as np
import pylab, random, math

'''
Available datasets:
randomdata
twoclusters
threeclusters
fourclusters
squareddata
'''
from messedupclusters import classA, classB, data, N

# kernel definitions
def linearKernel(x, y):
    return np.dot(x, y) + 1

def polynomialKernel(x, y, p = 2):
    return (np.dot(x, y) + 1) ** p

def radialBasisKernel(x, y, sigma = 3):
    return math.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

def sigmoidKernel(x, y, k = 2, delta = 1):
    return math.tanh(np.dot(k * x, y) - delta)

kernel = radialBasisKernel

def indicator(xnew, support):
    return sum([support[i][0] * support[i][2] * kernel(support[i][1], xnew) for i in range(len(support))])

'''
# kernel tests
a = np.array([1, 2, 3], float)
b = np.array([4, 5, 6], float)
print(linearKernel(a, b))
print(polynomialKernel(a, b, 2))
print(radialBasisKernel(a, b, 2))
print(sigmoidKernel(a, b, 0.1, 2))
'''

# build P matrix p(i, j) = ti tj K(xi, xj)
P = np.empty([N, N])
for i in range(0, N):
    for j in range(0, N):
        ti = data[i][2]
        tj = data[j][2]
        xi = np.array([data[i][0], data[i][1]])
        xj = np.array([data[j][0], data[j][1]])
        P[i, j] = ti * tj * kernel(xi, xj)

# build q, h, G
q = np.array([-1.0 for i in range(0, N)])
h = np.array([0.0  for i in range(0, N)])
G = np.zeros((N, N), float)
np.fill_diagonal(G, -1)

# solve convex problem
r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
alpha = list(r['x'])

# keep non zero alphas and their datapoints
# support[i][0] = alpha
# support[i][1] = datapoint numpy vector
# support[i][2] = datapoint class
support = []
for i in range(N):
    if alpha[i] > 10e-5:
        support.append([alpha[i], np.array([data[i][0], data[i][1]]), data[i][2]])

# draw contour lines
xrange = np.arange(-4, 4, 0.05)
yrange = np.arange(-4, 4, 0.05)
grid = matrix([
    [indicator(np.array([x, y], float), support) for y in yrange]
    for x in xrange
])
pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors = ('red', 'black', 'blue'), linewidths = (1, 3, 1))
pylab.hold(True)
pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')
pylab.show()