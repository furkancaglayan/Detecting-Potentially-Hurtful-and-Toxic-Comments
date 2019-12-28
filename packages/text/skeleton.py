from packages.text.textutilities import load_data


class Skeleton(object):
    def __init__(self):
        self.df_path = ""
        self.df = None
        self.column_name = ""
        self.text = None
        self.methods = []
        self.classifiers = []
        self.progress = ""
        self.data_size = 0

    def _create(self, dataframe_path, step, total_steps, related_column='comment_text'):
        self.df_path = dataframe_path
        self.df = load_data(dataframe_path, step, total_steps)
        self.column_name = related_column
        self.data_size = len(self.df)
        self.text = self.df[related_column]

    def build(self, methods, df_path):
        self.methods = methods
        self._create(df_path, 1, len(methods) + 1)
        for i, method in enumerate(methods):
            args = {'data': self.text, 'step': i + 2, 'total_steps': len(methods) + 1}
            self.text = method(**args)
        self.df[self.column_name] = self.text
        self.df.dropna(inplace=True, how="any")
        self.data_size = len(self.df)

    def classify(self, classifiers, keys: list):
        self.classifiers = classifiers

        X = self.text
        for i, clf in enumerate(self.classifiers):
            clf.keys = keys
            print("[Classifier {}/{}   Fitting data over {}...]".format(i + 1, len(self.classifiers), clf.name))
            self.progress += "[Classifier {}/{}   Fitting data over {}...]\n".format(i + 1, len(self.classifiers),
                                                                                     clf.name)
            for key in keys:
                y = self.df[key]
                clf.fit(X, y)
                clf.predict(key, X, self.df[key])
            self.progress += clf.debug()

    def save_progress(self, path):
        summary_out = open(path, "w")
        summary_out.write(self.progress)
        summary_out.close()
