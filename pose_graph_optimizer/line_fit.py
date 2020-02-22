from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 10, 100)
y = x**2

# ax = plt.subplot(1,1,1)
# ax.plot(x, y, 'k-')
# plt.show()

(slope, intercept, _, _, _) = stats.linregress(x,y)

x1 = 1; x2 =10
y1 = slope*x1 + intercept
y2 = slope*x2 + intercept

ax = plt.subplot(1,1,1)
ax.plot([x1, x2], [y1, y2], 'b-')
ax.plot(x, y, 'k-')
plt.show()