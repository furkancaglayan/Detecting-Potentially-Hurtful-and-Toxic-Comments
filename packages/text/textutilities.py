import pandas as pd


def load_data(data:str, step, total_steps, verbose=True):
    if verbose:
        print("[Reading File... Step {}/{}]\n".format(step, total_steps))
    try:
        df = pd.read_csv(data)
    except FileNotFoundError as e:
        print(e.strerror)
        return None
    return df


def clean_text(data: pd.DataFrame=None, step=0, total_steps=0, verbose=True):
    """
    Takes the related dataframe *column* and removes punctuation from it.
    :param step:
    :param total_steps:
    :param verbose:
    :param data:
    :return:
    """
    if verbose:
        print("[Cleaning data... Step {}/{}]".format(step, total_steps))
    # remove newline
    data = data.apply(lambda x: str(x).replace('\n', ' '))
    # replace i'm with i am
    data = data.apply(lambda x: str(x).replace('\'m', ' am'))
    # replace don't and isn't etc
    data = data.apply(lambda x: str(x).replace('n\'t', ' not'))
    # handle can't to can not
    data = data.apply(lambda x: str(x).replace('can\'t', 'can not'))
    # remove : from dates and time mostly
    data = data.apply(lambda x: str(x).replace(':', ''))
    # remove hashtag
    data = data.apply(lambda x: str(x).replace('#', ' '))

    data = data.apply(lambda x: str(x).replace('_', ' '))
    data = data.apply(lambda x: str(x).replace('\'s', ''))
    data = data.apply(lambda x: str(x).replace('-', ''))
    data = data.apply(lambda x: str(x).replace('"', ''))
    data = data.apply(lambda x: str(x).replace('\'', ''))
    data = data.apply(lambda x: str(x).replace('!', ''))
    data = data.apply(lambda x: str(x).replace('?', ''))
    data = data.apply(lambda x: str(x).replace('\\', ''))
    data = data.apply(lambda x: str(x).replace('/', ''))
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

    data = data.apply(lambda x: '' if type(x) is int or x is float else x)

    def remove_if_num(x: str) -> str:
        arr = x.split(' ')
        for i in range(len(arr)):
            if arr[i].isdigit():
                x = x.replace(arr[i], '')
        return x

    data = data.apply(lambda x: remove_if_num(x).strip())

    data = data.str.lower()
    if verbose:
        print("[Cleaning data is done! Step {}/{}]\n".format(step, total_steps))
    return data
