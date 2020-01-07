import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.text.textutilities import load_data
import time


class Skeleton(object):
    """
    Attributes
        ----------
        resource : str
            corpus path
        model : KeyedVectors
            Loaded word2vec model
        vocabulary : dict
            Vocabulary shortcut of the loaded model
        trained_model : KeyedVectors
            Trained word2vec model
    """

    def __init__(self, keys, random_state):
        self.df_path = ""
        self.df = None
        self.column_name = ""
        self.classifiers = []
        self.progress = ""
        self.data_size = 0
        self.start_time = time.time()
        self.utilities = None
        self.keys = keys
        self.random_state = random_state

    def _create(self, dataframe_path, related_column='comment_text'):
        self.df_path = dataframe_path
        self.df = load_data(dataframe_path)
        self.column_name = related_column
        self.data_size = len(self.df)

    def build(self, utilities, df_path):
        self.utilities = utilities
        self._create(df_path)
        for i, utility in enumerate(utilities):
            args = {'df': self.df, 'column': self.column_name}
            utility.what(i, len(utilities))
            self.df = utility.apply(**args)
        self.df.dropna(inplace=True, how="any")
        self.data_size = len(self.df)
        print("Skeleton build is done!")

    def classify(self, classifiers, X_x, X_y, y_x, y_y):
        self.classifiers = classifiers

        for i, clf in enumerate(self.classifiers):
            clf.keys = self.keys
            print("[Classifier {}/{}   Fitting data over {}...]".format(i + 1, len(self.classifiers), clf.name))
            self.progress += "[Classifier {}/{}   Fitting data over {}...]\n".format(i + 1, len(self.classifiers),
                                                                                     clf.name)
            for key in self.keys:
                clf.fit(X_x, X_y)
                clf.predict(key, y_x[key], y_y[key])
            self.progress += clf.debug()

    def save_progress(self, path):
        self.progress += "\n\n-----------End------------\nTime taken: {} seconds".format(time.time() - self.start_time)
        summary_out = open(path, "w")
        summary_out.write(self.progress)
        summary_out.close()

    def split_by_keys(self, n_category):
        df_splits = {}
        for each_key in self.keys:
            n = int(n_category / 2) if len(self.df[self.df[each_key] == 1]) >= n_category else int(
                len(self.df[self.df[each_key] == 1]) / 2)
            df_splits[each_key] = self.df[self.df[each_key] == 1].sample(n=n, random_state=self.random_state)
            non = (self.df[self.df[each_key] == 0].sample(n=n, random_state=self.random_state))
            df_splits[each_key] = df_splits[each_key].append(non, ignore_index=True)
        return df_splits

    def info(self):
        info = ""
        for key in self.keys:
            info += "{}: {} samples\n".format(key, len(self.df.loc[self.df[key] == 1]))
        # for i in range(len(self.keys[-1:]): if(self.keys[i] == 0 & self.keys[i+1] == 0 & self.keys[i+2] == 0 &
        # self.keys[i+3] == 0 & self.keys[i+4] == 0 & self.keys[i+5] == 0 & self.keys[i+6] == 0): info += "{}: {}
        # samples\n".format(self.keys[-1:], len(self.df.loc[self.df[self.keys[:-1]] == 0]))
        print(info)

    # def split_test_train(self,key, percentage=0.2, samples=None, random_state=20):
    #     ret_dict = {}
    #     df = samples[key]
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         ...
    #     X, y, test_size = 0.33, random_state = 42)
    #     test_indices=np.random.choice(int(len(df)*percentage))
    #     test = df.sample(frac=percentage, replace=False, random_state=random_state)
    #     df.drop(labels=test.index, inplace=True)
    #     ret_dict['test'] = test
    #     ret_dict['train'] = df
    #
    #     return ret_dict
