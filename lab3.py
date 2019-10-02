import sklearn
import numpy as np
import matplotlib.pyplot as plt 

data = np.array([[1,2],[2,3],[3,4],[4,5],[5,6]])
x = data[:,0]
y = data[:,1]

plt.scatter(x,y)
plt.grid(True)
plt.show()