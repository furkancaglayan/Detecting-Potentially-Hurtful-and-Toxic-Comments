import numpy as np
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


class _Classifier(object):
    """
    Base classifier class that implements sklearn models inside.

    Attributes
        ----------
        estimator : Sklearn model
            sklearn model that implements fit and predict
        name : str
            Name of the classifier
        keys : list
            List of classes
        accuracies : dict
            Accuracy dict per each class
        predictions : dict
            Predictions dict per each class
        correct_classifications : dict
            Number of correct classifications for each class
        test_sizes : dict
            Length of each test data size per each class
    """

    def __init__(self, name, keys=None):
        """
        :param name: Name of the classifier
        :param keys: Classes list
        """
        self.estimator = None
        self.name = name
        self.keys = keys
        self.accuracies = {}
        self.predictions = {}
        self.correct_classifications = {}
        self.test_sizes = {}

    def fit(self, X, y):
        """
        Abstraction of sklearn's fit.

        :param X: data
        :param y: labels
        :return:
        """
        self.estimator.fit(X, y)

    def predict(self, key, test_X, test_y):
        """
        Prediction abstraction of sklearn's predict

        :param key: class name
        :param test_X: test data
        :param test_y: test label
        :return:
        """
        self.correct_classifications[key] = np.sum(self.predictions[key] == test_y)
        self.accuracies[key] = self.correct_classifications[key] / len(test_y)
        self.test_sizes[key] = len(test_X)

    def debug(self, everything=True):
        """
        Prints and returns test details.
        :param everything:
        :return:
        """
        summary = ""
        for i, key in enumerate(self.keys):
            summary += "  Category #{}: {} \n".format(i + 1, key)
            summary += "    Accuracy is {}\n".format(self.accuracies[key])
            if everything:
                summary += "    Total correct classifications is {}\n".format(
                    self.correct_classifications[
                        key])
                summary += "    Total test data size is {}\n".format(self.test_sizes[key])

        print(summary)
        return summary + "\n"


class _NonEnsembleClassifier(_Classifier):
    """
    Base class for the simple sklearn models, Decision Tree, Naive Bayes and SVM
    """

    def __init__(self, name):
        super().__init__(name)

    def predict(self, key, test_X, test_y):
        self.predictions[key] = self.estimator.predict(test_X)
        super().predict(key, test_X, test_y)


class DecisionTree(_NonEnsembleClassifier):
    def __init__(self):
        super().__init__("Decision Tree")
        self.estimator = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)),
            ('clf', OneVsRestClassifier(DecisionTreeClassifier(max_depth=4))),
        ], verbose=False)

    def predict(self, key, test_X, test_y):
        self.predictions[key] = self.estimator.predict(test_X)
        super().predict(key, test_X, test_y)


class NaiveBayes(_NonEnsembleClassifier):
    def __init__(self):
        super().__init__("Naive Bayes")
        self.estimator = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)),
            ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
        ], verbose=False)

    def predict(self, key, test_X, test_y):
        self.predictions[key] = self.estimator.predict(test_X)
        super().predict(key, test_X, test_y)


class SVM(_NonEnsembleClassifier):
    def __init__(self):
        super().__init__("SVM")
        self.estimator = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)),
            ('clf', OneVsRestClassifier(SVC(gamma='auto'))),
        ], verbose=False)

    def predict(self, key, test_X, test_y):
        self.predictions[key] = self.estimator.predict(test_X)
        super().predict(key, test_X, test_y)


class AveragingEstimator(_Classifier):
    """
    Given 3 estimators(Decision Tree, Naive Bayes, SVM), returns accuracy average of them.
    """

    def __init__(self):
        super().__init__("Average of SVM, Naive Bayes and Decision Tree")
        self.estimators = [DecisionTree(), NaiveBayes(), SVM()]

    def fit(self, X, y):
        for clf in self.estimators:
            clf.fit(X, y)

    def predict(self, key, test_X, test_y):
        for clf in self.estimators:
            clf.predict(key, test_X, test_y)
        self.predictions[key] = np.rint(np.average(
            self.estimators[0].predictions[key] + self.estimators[1].predictions[key] + self.estimators[2].predictions[
                key]))
        super().predict(key, test_X, test_y)


class _AdaBoostClassifier(_Classifier):
    """
    Base class that uses AdaBoost algorithm with 3 base estimators.
    """

    def __init__(self, name, estimator):
        super().__init__(name)
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)

    def predict(self, key, test_X, test_y):
        self.predictions[key] = self.estimator.predict(test_X)
        super().predict(key, test_X, test_y)


class AdaBoostDecisionTree(_AdaBoostClassifier):
    def __init__(self, max_tree_depth=8, n_estimators=100):
        clf_base = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=max_tree_depth),
            n_estimators=n_estimators, learning_rate=0.5
        )
        clf = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)),
            ('clf', clf_base),
        ], verbose=False)
        super().__init__("AdaBoost Classifier, Decision Tree with {} estimators".format(n_estimators), clf)


class AdaBoostNaiveBayes(_AdaBoostClassifier):
    def __init__(self, n_estimators=100):
        clf_base = AdaBoostClassifier(
            MultinomialNB(fit_prior=True, class_prior=None),
            n_estimators=n_estimators, learning_rate=0.5
        )
        clf = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)),
            ('clf', clf_base),
        ], verbose=False)
        super().__init__("AdaBoost Classifier, MN Naive Bayes with {} estimators".format(n_estimators), clf)


class AdaBoostSVM(_AdaBoostClassifier):
    def __init__(self, n_estimators=100):
        self.estimator = SVC(gamma='auto')
        super().__init__("AdaBoost Classifier, SVM with {} estimators".format(n_estimators), n_estimators)

# class BoostingDecisionTree(_Classifier): def __init__(self, base_estimator: _Classifier): super().__init__(
# "AdaBoost Classifier " + base_estimator.name) self.estimator = base_estimator self.ensemble = AdaBoostClassifier(
# base_estimator=self.estimator.estimator, n_estimators=100, algorithm="SAMME")
#
#     def fit(self, X, y):
#         self.ensemble.fit(X, y)
#
#     def predict(self, key, test_X, test_y):
#         self.predictions[key] = self.ensemble.predict(test_X)
#         super().predict(key, test_X, test_y)
