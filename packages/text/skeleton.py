from packages.text.textutilities import load_data
import time


class Skeleton(object):
    def __init__(self):
        self.df_path = ""
        self.df = None
        self.column_name1 = ""
        self.column_name2 = ""
        self.column_name3 = ""
        self.column_name4 = ""
        self.column_name5 = ""
        self.column_name6 = ""
        self.column_name7 = ""

        self.text = None
        self.tags = None
        self.methods = []
        self.classifiers = []
        self.progress = ""
        self.data_size = 0
        self.start_time = time.time()

    def _create(self, dataframe_path, step, total_steps,
                related_column1='comment_text', related_column2 = 'toxic',
                related_column3 = 'severe_toxic', related_column4 = 'obscene', related_column5 = 'threat',
                related_column6 = 'insult', related_column7 = 'identity_hate'
               ):
        
        self.df_path = dataframe_path
        self.df = load_data(dataframe_path, step, total_steps)
        self.column_name1 = related_column1
        self.column_name2 = related_column2
        self.column_name3 = related_column3
        self.column_name4 = related_column4
        self.column_name5 = related_column5
        self.column_name6 = related_column6
        self.column_name7 = related_column7
        self.data_size = len(self.df)
        self.text = self.df[related_column1,related_column2,related_column3,related_column4,related_column5,related_column6,related_column7]
        self.tags = self.df[related_column2,related_column3,related_column4,related_column5,related_column6,related_column7]

    def build(self, methods, df_path):
        self.methods = methods
        self._create(df_path, 1, len(methods) + 1)
        for i, method in enumerate(methods):
            args = {'data': self.text, 'step': i + 2, 'total_steps': len(methods) + 1}
            self.text = method(**args)
        self.df[self.column_name1] = self.text
        self.df[self.column_name2,self.column_name3,self.column_name4,self.column_name5,self.column_name6,self.column_name7 ] = self.tags
        
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
        self.progress+="\n\n-----------End------------\nTime taken: {} seconds".format(time.time()-self.start_time)
        summary_out = open(path, "w")
        summary_out.write(self.progress)
        summary_out.close()
