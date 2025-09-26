import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

data = {
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temperature": [85,80,83,70,68,65,64,72,69,75,75,72,81,71],
    "Windy": [False,True,False,False,False,True,True,False,False,False,True,True,False,True],
    "Play": ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]
}

df = pd.DataFrame(data)

enc = OneHotEncoder(sparse_output=False)
out_enc = enc.fit_transform(df[["Outlook"]])
wind = df["Windy"].astype(int).values.reshape(-1,1)
X = np.hstack([df[["Temperature"]].values, out_enc, wind])
y = (df["Play"] == "Yes").astype(int).values

model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

feature_names = ["Temperature"] + list(enc.get_feature_names_out(["Outlook"])) + ["Windy"]

plt.figure(figsize=(8,5))
plot_tree(model, feature_names=feature_names, filled=True, impurity=False, class_names=None)
plt.show()

