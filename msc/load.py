import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

fig = plt.figure()
ax = plt.axes()

x = np.array([0.3212098797,0.4074196876,0.4956667444])
y = np.array([76.7199,78.1222,79.0306])
plt.plot(x, y)
plt.savefig('plot_basket.png')