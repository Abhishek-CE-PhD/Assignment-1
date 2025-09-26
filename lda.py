import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# example dataset
X = np.array([[4,2],[2,4],[2,3],[3,6],[4,4],
              [9,10],[6,8],[9,5],[8,7],[10,8]])
y = np.array([0,0,0,0,0,1,1,1,1,1])

lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)

print("Transformed data:\n", X_lda)

# plot
plt.scatter(X_lda[y==0], np.zeros_like(X_lda[y==0]), label="Class 0")
plt.scatter(X_lda[y==1], np.zeros_like(X_lda[y==1]), label="Class 1")
plt.xlabel("LDA Component 1")
plt.legend()
plt.title("LDA Projection")
plt.show()
