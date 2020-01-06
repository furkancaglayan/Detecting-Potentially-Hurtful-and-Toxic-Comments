import multiprocessing

import gensim
import nltk
import pandas
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases
from nltk.data import find

import packages.text.textutilities as utilities
from packages.augmentation.random import RandomMachine


class EmbeddingAugmentation(object):
    """
        A class to augment text data.
        ...

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

    def __init__(self, load_path=None):
        """

        :param load_path: Path of the custom word2vec model. Leave as None if
        you want to use pre-trained model from NLTK.
        """
        if load_path is None:
            nltk.download('word2vec_sample')
            self.resource = str(find('models/word2vec_sample/pruned.word2vec.txt'))
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.resource, binary=False)
        else:
            print("Loading Custom Model!")
            self.model = Word2Vec.load(load_path)

        self.vocabulary = self.model.wv.vocab
        self.trained_model = None

    def get_similar_words(self, word, n=3):
        """
        Finds and returns similar words from the model.

        :param word: a single word to get similar words to.
        :param n: Count of similar words to return.
        :return: List of similar words. If can't find similar words, returns None

        Raises
        ------
        :raises:
            KeyError: If word is not included in the model vocab.
        """
        try:
            similar = self.model.most_similar(positive=[word], topn=n)
            word_list = []
            for w, _ in similar:
                word_list.append(str(w).lower())
            return word_list

        except KeyError as err:
            print(err)
            return None

    def get_similar_words_trained(self, word, n=3):
        """
        Finds and returns similar words from the trained model. Must complete train function first.

        :param word: a single word to get similar words to.
        :param n: Count of similar words to return.
        :return: List of similar words. If can't find similar words, returns None

        Raises
        ------
        :raises:
            KeyError: If word is not included in the model vocab.
        """
        if self.trained_model is None:
            print("Train the model first.")
            return None
        try:
            similar = self.trained_model.wv.most_similar(positive=[word], topn=n)
            word_list = []

            for w, _ in similar:
                word_list.append(str(w).lower())

            return word_list

        except KeyError as key_error:
            print(key_error)
            return None

    def train(self, data, size=100):
        """
        Trains a separate new word2vec model.

        :param data: Data to train on.
        :param size: Size of the vocab.
        :return:
        """
        print("[Training a new model from the data]")
        data = [row.split() for row in data]
        phrases = Phrases(data, min_count=30, progress_per=10000)
        sentences = phrases[data]
        self.trained_model = gensim.models.Word2Vec(min_count=20,
                                                    window=2,
                                                    size=size,
                                                    sample=6e-5,
                                                    alpha=0.03,
                                                    min_alpha=0.0007,
                                                    negative=20,
                                                    workers=multiprocessing.cpu_count() - 1)
        self.trained_model.build_vocab(sentences, progress_per=10000)
        print("[Built Vocab]")
        self.trained_model.train(sentences, total_examples=self.trained_model.corpus_count, epochs=30, report_delay=1)
        print("[Training is done]")

    def save_trained_model(self, name="word2vec"):
        """
        Exports the trained model. Must complete train function first.

        :param name: name of the model to save. Exclude .model extension.
        :return:
        """
        if self.trained_model is None:
            print("[Model is empty]")
            return
        self.trained_model.save("{}.model".format(name))

    def augment(self, sentence: str, random_state=20):
        """
        For a given sentence, returns a single augmented sentence.
        Algorithms works like this:
        1 - Determine a base chance(see random.py for more details)
        2 - For each augmentaeble word, run RandomMachine.pass_chance()
        3 - If RandomMachine returns False, it means this word will stay as the same.
            But next word's chance of being augmented will increase.
        4 - If RandomMachine returns True, it means word will be augmented. Next word's
            chance of being augmented is reset to base chance.


        :param sentence: Sentence to augment
        :param random_state: Uses predetermined random state for reproducibility purposes.
        :return: augmented sentence
        """
        randomizer = RandomMachine(random_state=random_state)
        augmentaeble_word_count = self._get_augmentable_word_count(sentence)
        if augmentaeble_word_count == 0:
            return None

        final_augment = ""
        for word in sentence.split(' '):
            if randomizer.pass_chance():
                similar = self.get_similar_words(word, n=5)
                if similar is None:
                    final_augment += word + " "
                    continue
                replacement_word = similar[randomizer.gen_random_int(0, len(similar))]
                final_augment += replacement_word + " "
            else:
                final_augment += word + " "
        return utilities.TextCleaner.clean_sample(final_augment)

    def _get_augmentable_word_count(self, sentence) -> int:
        """
        Finds how many words in a sentence is augmentable and returns it.

        :param sentence:
        :return:
        """
        n = 0
        for word in sentence.split(' '):
            try:
                self.model.wv.most_similar(word)
                n += 1
            except KeyError as err:
                continue
        return n

    def populate(self, keys, data, target, random, augmentation_per_sentence, text_id="comment_text"):

        df_list = []
        for key in keys:
            df = data[key]
            diff = target - len(df)
            # select sentences to augment
            if diff > 0:
                ind = 0
                new_list = []
                random_comment_df = df.sample(n=diff, random_state=random)
                while diff:
                    if ind == diff:
                        # zero the ind
                        ind -= ind
                    comment = random_comment_df[ind]
                    augmented_comment = self.augment_single(comment)
                    df_list.append(comment)
                    df_list.append(augmented_comment)
                    diff -= 1
                    ind += 1

            else:
                random_comment_df = df.sample(n=target, random_state=random)

            new_df = pandas.DataFrame(df_list)
            data[key] = new_df
