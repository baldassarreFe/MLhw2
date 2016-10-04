import random

'''Random data creation'''
classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(15)] + \
         [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(15)]
classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(30)]

data = classA + classB
random.shuffle(data)
N = len(data)
