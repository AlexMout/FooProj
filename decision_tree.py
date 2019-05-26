from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree.export import export_text
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
import pdb
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus
import collections

def save_tree_png(classifier,columns, title:str="tree.png"):
	# Visualize data
	dot_data = tree.export_graphviz(classifier,
	                                feature_names=df_X.columns,
	                                out_file=None,
	                                filled=True,
	                                rounded=True)
	graph = pydotplus.graph_from_dot_data(dot_data)

	colors = ('turquoise', 'orange')
	edges = collections.defaultdict(list)

	for edge in graph.get_edge_list():
	    edges[edge.get_source()].append(int(edge.get_destination()))

	for edge in edges:
	    edges[edge].sort()    
	    for i in range(2):
	        dest = graph.get_node(str(edges[edge][i]))[0]
	        dest.set_fillcolor(colors[i])

	graph.write_png(title)

if __name__ == "__main__":
	sns.set(color_codes=True)

	iris = load_iris()

	X = iris["data"]
	Y = iris["target"]

	df_X = pd.DataFrame(X,columns=iris["feature_names"])

	df_Y = pd.DataFrame(Y,columns=["Target"])

	dico= {index:elem for index,elem in enumerate(iris["target_names"])}

	df_Y["Target"] = df_Y["Target"].map(dico)

	df = pd.concat([df_X,df_Y],axis=1)

	sns.pairplot(df,hue="Target")

	plt.show()

	sns.heatmap(df.corr(),xticklabels=df_X.columns, yticklabels=df_X.columns,annot=True)
	plt.show()

	# Separating dataset into train & test set
	x_train , x_test, y_train, y_test = train_test_split(df_X, df_Y, test_size = 0.25, random_state = 0)

	# create and train the classifier
	classifier = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=4)
	classifier.fit(x_train, y_train)

	# try predictions on the test set
	y_pred = classifier.predict(x_test)

	cm = confusion_matrix(y_test, y_pred) # rows : original class / columns : predicted class
	print(cm)
	accuracy = sum(cm[i][i] for i in range(3)) / y_test.shape[0]
	print("accuracy = " + str(accuracy))


	save_tree_png(classifier, df_X.columns)


