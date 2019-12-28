import random

import pandas as pd


def load_data(data: str, step, total_steps):
    print("[Step {}/{}   Reading File...]".format(step, total_steps))
    try:
        df = pd.read_csv(data)
        return df
    except FileNotFoundError as e:
        print(e.strerror)
        return None


def sample(data: pd.DataFrame = None, key_name='comment_text', step=0, total_steps=0, n=5):
    print("[Step {}/{}   Sampling data...]\n".format(step, total_steps))
    l = len(data)
    for i in range(n):
        random_int = int(random.randrange(0, l - 1))
        row = data.iloc[random_int]
        print("Sample {}: \n".format(i + 1) + row)
    return data


def trim_corpus(data: pd.DataFrame = None, key_name='comment_text', step=0, total_steps=0, n=25):
    print("[Step {}/{}   Trimming data...]\n".format(step, total_steps))
    for i, row in enumerate(data):
        if len(row.split(' ')) <= n:
            data.iloc[i] = None
    return data


def clean_text(data: pd.DataFrame = None, step=0, total_steps=0):
    """
    Takes the related dataframe *column* and removes punctuation from it.
    :param step:
    :param total_steps:
    :param data:
    :return:
    """
    print("[Step {}/{}   Cleaning data...]".format(step, total_steps))
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
    return data


def clean_sample(txt="You are mentioned in Wikipedia:Arbitration/Requests/Clarification#Request_for_clarification"
                     ":_Wikipedia:Arbitration/Requests/Case/Abd-William_M._Connolley. Thought you'd like to know. "
                     "Happy New Year, may it surpass the old!"):
    print(txt)
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

    print(txt.lower().strip())
