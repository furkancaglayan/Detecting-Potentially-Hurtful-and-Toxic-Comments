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
    def __init__(self, name, keys=None):
        self.estimator = None
        self.name = name

        self.keys = keys
        self.accuracies = {}
        self.predictions = {}
        self.correct_classifications = {}
        self.test_sizes = {}

    def fit(self, X, y):
        self.estimator.fit(X, y)

    def predict(self, key, test_X, test_y):
        self.correct_classifications[key] = np.sum(self.predictions[key] == test_y)
        self.accuracies[key] = self.correct_classifications[key] / len(test_y)
        self.test_sizes[key] = len(test_X)

    def debug(self, everything=True):
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


class NaiveBayes(_NonEnsembleClassifier):
    def __init__(self):
        super().__init__("Naive Bayes")
        self.estimator = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)),
            ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
        ], verbose=False)


class SVM(_NonEnsembleClassifier):
    def __init__(self):
        super().__init__("SVM")
        self.estimator = Pipeline([
            ('vectorizer', TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)),
            ('clf', OneVsRestClassifier(SVC(gamma='auto'))),
        ], verbose=False)


class AveragingEstimator(_Classifier):
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
    def __init__(self, name, n_estimators):
        super().__init__(name)
        self.ensemble = AdaBoostClassifier(base_estimator=self.estimator, n_estimators=n_estimators, algorithm="SAMME")

    def fit(self, X, y):
        self.ensemble.fit(TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS).fit_transform(X), y)

    def predict(self, key, test_X, test_y):
        self.predictions[key] = self.ensemble.predict(TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS).fit_transform(
            test_X))
        super().predict(key, test_X, test_y)


class AdaBoostDecisionTree(_AdaBoostClassifier):
    def __init__(self, max_tree_depth=4, n_estimators=100):
        self.estimator = DecisionTreeClassifier(max_depth=max_tree_depth)
        super().__init__("AdaBoost Classifier, Decision Tree with {} estimators".format(n_estimators), n_estimators)


class AdaBoostNaiveBayes(_AdaBoostClassifier):
    def __init__(self, n_estimators=100):
        self.estimator = MultinomialNB(fit_prior=True, class_prior=None)
        super().__init__("AdaBoost Classifier, MN Naive Bayes with {} estimators".format(n_estimators),n_estimators)


class AdaBoostSVM(_AdaBoostClassifier):
    def __init__(self, n_estimators=100):
        self.estimator = SVC(gamma='auto')
        super().__init__("AdaBoost Classifier, SVM with {} estimators".format(n_estimators),n_estimators)

