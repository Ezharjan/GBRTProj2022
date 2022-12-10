from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from sklearn.metrics import accuracy_score
iris_sample = load_iris()
x = iris_sample.data
y = iris_sample.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=123)
svclf = svm.SVC(kernel='rbf', decision_function_shape='ovr', random_state=123)
treeclf = DecisionTreeClassifier()
gbdtclf = GradientBoostingClassifier(learning_rate=0.7)
lrclf = LogisticRegression()
scclf = StackingCVClassifier(
    classifiers=[svclf, treeclf, gbdtclf], meta_classifier=lrclf, cv=5)
scclf.fit(x_train, y_train)
scclf_pre = scclf.predict(x_test)
print('真实值：', y_test)
print('预测值：', scclf_pre)
print('准确度：', accuracy_score(scclf_pre, y_test))