from sklearn import tree

# Training input
# 1=smooth, 0=bumpy
features = [
    [140, 1],
    [130, 1],
    [150, 0],
    [170, 0]
]

# Training output
# 0=apple, 1=orange
labels = [0, 0, 1, 1]

# Decision tree of rules
clf = tree.DecisionTreeClassifier()
# Fit = training algo
clf = clf.fit(features, labels)
# Input and output type are always the same

print clf.predict([[160, 0]]) # => [1], ie orange
