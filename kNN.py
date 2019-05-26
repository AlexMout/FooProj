from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
import pdb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris["data"]

Y = iris["target"]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=.75)

df_x = pd.DataFrame(x_test,columns=iris["feature_names"])

classifier = KNeighborsClassifier()

classifier.fit(x_train,y_train)

predictions = classifier.predict(x_test)

print(accuracy_score(y_test, predictions))

pred = pd.DataFrame(predictions,columns=["prediction"])

df_x["wellClassified"] = pred["prediction"] == y_test

dico_mapping = {
	True:"Good",
	False:"Wrong"
}

df_x["wellClassified"] = df_x["wellClassified"].map(dico_mapping)

# Visualize the good and wrong predictions
sns.pairplot(df_x,hue="wellClassified")
plt.show()


pdb.set_trace()

