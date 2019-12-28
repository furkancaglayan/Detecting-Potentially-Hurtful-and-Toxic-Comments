import pandas as pd


def load_data(path: str):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        print(e.strerror)
        return None
    return df


def clean_text(df: pd.DataFrame, lowercase=True):
    """
    Takes the related dataframe *column* and removes punctuation from it.
    :param df:
    :param lowercase:
    :return:
    """

    # remove newline
    df = df.apply(lambda x: str(x).replace('\n', ' '))
    # replace i'm with i am
    df = df.apply(lambda x: str(x).replace('\'m', ' am'))
    # replace don't and isn't etc
    df = df.apply(lambda x: str(x).replace('n\'t', ' not'))
    # handle can't to can not
    df = df.apply(lambda x: str(x).replace('can\'t', 'can not'))
    # remove : from dates and time mostly
    df = df.apply(lambda x: str(x).replace(':', ''))
    # remove hashtag
    df = df.apply(lambda x: str(x).replace('#', ' '))

    df = df.apply(lambda x: str(x).replace('_', ' '))
    df = df.apply(lambda x: str(x).replace('\'s', ''))
    df = df.apply(lambda x: str(x).replace('-', ''))
    df = df.apply(lambda x: str(x).replace('"', ''))
    df = df.apply(lambda x: str(x).replace('\'', ''))
    df = df.apply(lambda x: str(x).replace('!', ''))
    df = df.apply(lambda x: str(x).replace('?', ''))
    df = df.apply(lambda x: str(x).replace('\\', ''))
    df = df.apply(lambda x: str(x).replace('/', ''))
    df = df.apply(lambda x: str(x).replace('[', ''))
    df = df.apply(lambda x: str(x).replace(']', ''))
    df = df.apply(lambda x: str(x).replace(')', ''))
    df = df.apply(lambda x: str(x).replace('(', ''))
    df = df.apply(lambda x: str(x).replace('{', ''))
    df = df.apply(lambda x: str(x).replace('}', ''))
    df = df.apply(lambda x: str(x).replace('.', ''))
    df = df.apply(lambda x: str(x).replace(',', ''))
    df = df.apply(lambda x: str(x).replace('•', ''))
    df = df.apply(lambda x: str(x).replace('   ', ' '))
    df = df.apply(lambda x: str(x).replace('  ', ' '))

    df = df.apply(lambda x: '' if type(x) is int or x is float else x)

    print("Removing numbers...")

    def remove_if_num(x: str) -> str:
        arr = x.split(' ')
        for i in range(len(arr)):
            if arr[i].isdigit():
                x = x.replace(arr[i], '')
        return x

    df = df.apply(lambda x: remove_if_num(x).strip())
    if lowercase:
        df = df.str.lower()
    return df
