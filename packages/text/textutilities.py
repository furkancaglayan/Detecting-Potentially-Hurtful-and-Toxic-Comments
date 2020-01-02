import random
import pandas as pd


class _Utility(object):
    def __init__(self):
        pass

    def apply(self, df, column):
        pass

    def what(self, index, size):
        pass


class Sampler(_Utility):
    def __init__(self, _min=2, _max=5):
        self._min = _min
        self._max = _max
        super().__init__()

    def print_samples(self, df, column):
        n = int(random.randrange(self._min, self._max))
        l = len(df)
        for i in range(n):
            random_int = int(random.randrange(0, l - 1))
            row = df[column].iloc[random_int]
            if row is None:
                continue
            print("Sample {}: \n".format(i + 1) + row)
        return df

    def apply(self, df, column):
        self.print_samples(df, column)
        return df

    def what(self, index, size):
        print("[Step {}/{}   Sampling data...]".format(index + 1, size))


class TextCleaner(_Utility):
    def __init__(self):
        super().__init__()

    def apply(self, df, column):
        return self.clean_text(df, column)

    def clean_text(self, df, column):
        data = df[column]
        # remove newline
        data = data.apply(lambda x: str(x).replace('\n', ' '))
        # replace i'm with i am
        data = data.apply(lambda x: str(x).replace('\'m', ' am'))
        # replace don't and isn't etc
        data = data.apply(lambda x: str(x).replace('n\'t', ' not'))
        # handle can't to can not
        data = data.apply(lambda x: str(x).replace('can\'t', 'can not'))
        data = data.apply(lambda x: str(x).replace('\'d', ' would'))
        # remove : from dates and time mostly
        data = data.apply(lambda x: str(x).replace(':', ' '))
        # remove hashtag
        data = data.apply(lambda x: str(x).replace('#', ' '))
        data = data.apply(lambda x: str(x).replace('_', ' '))
        data = data.apply(lambda x: str(x).replace('\'s', ''))
        data = data.apply(lambda x: str(x).replace('-', ' '))
        data = data.apply(lambda x: str(x).replace('"', ''))
        data = data.apply(lambda x: str(x).replace('\'', ' '))
        data = data.apply(lambda x: str(x).replace('|', ' '))
        data = data.apply(lambda x: str(x).replace('!', ''))
        data = data.apply(lambda x: str(x).replace('?', ''))
        data = data.apply(lambda x: str(x).replace('\\', ''))
        data = data.apply(lambda x: str(x).replace('/', ' '))
        data = data.apply(lambda x: str(x).replace('[', ''))
        data = data.apply(lambda x: str(x).replace(']', ''))
        data = data.apply(lambda x: str(x).replace(')', ''))
        data = data.apply(lambda x: str(x).replace('(', ''))
        data = data.apply(lambda x: str(x).replace('{', ''))
        data = data.apply(lambda x: str(x).replace('}', ''))
        data = data.apply(lambda x: str(x).replace('.', ''))
        data = data.apply(lambda x: str(x).replace(',', ''))
        data = data.apply(lambda x: str(x).replace('•', ''))
        data = data.apply(lambda x: str(x).replace('   ', ' '))
        data = data.apply(lambda x: str(x).replace('  ', ' '))
        data = data.apply(lambda x: str(x).replace('=', ''))
        data = data.apply(lambda x: str(x).replace('+', ''))

        data = data.apply(lambda x: '' if type(x) is int or x is float else x)

        def remove_if_num(x: str) -> str:
            arr = x.split(' ')
            for i in range(len(arr)):
                if arr[i].isdigit():
                    x = x.replace(arr[i], '')
            return x

        data = data.apply(lambda x: remove_if_num(x).strip())

        data = data.str.lower()
        df[column] = data
        return df

    @staticmethod
    def clean_sample(txt="Sample Text"):
        txt = str(txt).replace('\n', ' ')
        txt = str(txt).replace('\'m', ' am')
        txt = txt.replace('n\'t', ' not')
        txt = txt.replace('can\'t', 'can not')
        txt = txt.replace('\'d', ' would')
        txt = txt.replace(':', ' ')
        txt = txt.replace('#', ' ')
        txt = txt.replace('_', ' ')
        txt = txt.replace('\'s', '')
        txt = txt.replace('-', ' ')
        txt = txt.replace('"', '')
        txt = txt.replace('\'', ' ')
        txt = txt.replace('!', '')
        txt = txt.replace('?', '')
        txt = txt.replace('/', ' ')
        txt = txt.replace('\'', '')
        txt = txt.replace('[', '')
        txt = txt.replace(']', '')
        txt = txt.replace('{', '')
        txt = txt.replace('}', '')
        txt = txt.replace('(', '')
        txt = txt.replace(')', '')
        txt = txt.replace('.', '')
        txt = txt.replace(',', '')
        txt = txt.replace('•', '')
        txt = txt.replace('   ', ' ')
        txt = txt.replace('  ', ' ')
        txt = txt.replace('=', '')
        txt = txt.replace('+', '')
        txt = txt.replace('|', ' ')

        return txt.lower().strip()

    def what(self, index, size):
        print("[Step {}/{}   Cleaning data...]".format(index + 1, size))


class Trimmer(_Utility):
    def __init__(self, threshold=10):
        self.threshold = threshold
        super().__init__()

    def trim_corpus(self, df, column):
        data = df[column]
        for i, row in enumerate(data):
            if len(row.split(' ')) <= self.threshold:
                data.iloc[i] = None
        return data

    def apply(self, df, column):
        text = self.trim_corpus(df, column)
        df[column] = text
        return df

    def what(self, index, size):
        print("[Step {}/{}  Trimming data...]".format(index + 1, size))


def load_data(data: str):
    try:
        df = pd.read_csv(data)
        return df
    except FileNotFoundError as e:
        print(e.strerror)
        return None
