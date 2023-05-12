import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

plt.scatter(zp, ks)
plt.show()

model = LinearRegression()
zp = zp.reshape(-1, 1)

model.fit(zp, ks)
LinearRegression()

model.intercept_, model.coef_
print(model.intercept_, model.coef_)

plt.scatter(zp, ks)
plt.plot(ks, 444.18+2.62*ks)
plt.show()

