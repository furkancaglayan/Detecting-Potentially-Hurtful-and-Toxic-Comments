import pandas as pd


class Skeleton(object):
    def __init__(self, dataframe_path, related_column='comment_text', verbose=True):
        self.df_path = dataframe_path
        self.df = pd.read_csv(dataframe_path)
        self.column_name = related_column
        self.text = self.df[related_column]
        self.verbose = verbose
        self.methods = []

    def build(self, methods):
        self.methods = methods
        for i, method in enumerate(methods):
            print(method)
            args = {'data': self.text, 'step': i + 1, 'total_steps': len(methods+1), 'verbose': True}
            self.text = method(**args)
        self.df[self.column_name] = self.text
