import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


X = 2 * np.random.rand(200, 1)
y = 4 + 5 * X + np.random.randn(200, 1)

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X, y, s=10)
plt.plot(X, y_pred, color="red")
plt.show()
