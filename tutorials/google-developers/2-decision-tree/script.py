import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
iris = load_iris()

# print iris.feature_names # Item input
# print iris.target_names # Label output
# print iris.data[0]
# print iris.target[0]

# Separate the data and use it for later testing
test_idx = [0, 50, 100]

# Training data
# Datasets minus the data we want for later testing
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)

# Viz code to generate PDF
from sklearn.externals.six import StringIO
# brew install graphviz
import pydotplus # Instead of pydot in video
dot_data = StringIO()
tree.export_graphviz(clf,
    out_file=dot_data,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True, rounded=True,
    impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
