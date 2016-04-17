from sklearn import tree
# we are gonna train and learn to discriminate apples and oranges
features = [[140,1], [130,1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
# DecisionTreeClassifier is just a classifier we need to
# obtain a learning algorithm
clf = tree.DecisionTreeClassifier()
# in scikit training algorithm is included in the classifier object
# fit is a synonym for "finding patterns in data"
clf = clf.fit(features, labels)
# now we have trained classifier
