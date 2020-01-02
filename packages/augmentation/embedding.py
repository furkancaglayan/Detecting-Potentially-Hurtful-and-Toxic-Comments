import gensim
import nltk
from gensim.models import Word2Vec
from nltk.data import find
import packages.text.textutilities as utilities
from gensim.models.phrases import Phrases, Phraser
import multiprocessing


class EmbeddingAugmentation(object):
    def __init__(self, load_path=None):
        if load_path is None:
            nltk.download('word2vec_sample')
            self.resource = str(find('models/word2vec_sample/pruned.word2vec.txt'))
            self.model = gensim.models.KeyedVectors.load_word2vec_format(self.resource, binary=False)
        else:
            self.model = Word2Vec.load(load_path)

        self.vocabulary = self.model.wv.vocab
        self.trained_model = None

    def get_similar_words(self, word, n=3):
        try:
            similar = self.model.most_similar(positive=[word], topn=n)
            word_list = []

            for w, _ in similar:
                word_list.append(str(w).lower())

            return word_list

        except KeyError as key_error:
            print(key_error)
            return None

    def get_similar_words_trained(self, word, n=3):
        """
        Get n similar words from the newly trained model.
        :param word: Word to find similar words to.
        :param n: Size of similar words.
        :return:
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

    def train_new(self, data, size=100):
        """
        Train a separate new word2vec model.
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
        if self.trained_model is None:
            print("[Model is empty]")
            return
        self.trained_model.save("{}.model".format(name))

    def replace_sentence(self, sentence: str, n=1):
        try:
            ret = []
            words = sentence.split(' ')
            l = len(words)

            for i in range(n):
                augmented_sentence=""
                for j in range(l):
                    pass
                
        except Error as error:
            print(error)
            return None


    def populate(self, keys, data, target, random, augmentation_per_sentence, text_id="comment_text"):
        for key in keys:
            df = data[key]
            diff = target - len(df)
            # select sentences to augment
            if(diff>0):
                random_comment_selections = df.sample(n=diff, random_state=random)
                for row in random_comment_selections:
                    text = row[text_id]
                    augmentations = self.replace_sentence(text, n=diff)
            else if(diff<0):
                df = df[:target]
