import numpy as np


class RandomMachine(object):
    """
    Random number generator for augmentation phase
    """

    def __init__(self, base_chance=0.5, step=0.1, random_state=20):
        """

        :param base_chance: Base chance of a word being augmented. See :func:`<embedding.EmbeddingAugmentation.augment>'
        :param step: Increase in chance if pass_chance returns False
        :param random_state:
        """
        self.base_chance = base_chance
        self.run_chance = base_chance
        self.step = step

        np.random.seed(random_state)

    def pass_chance(self) -> bool:
        """
        A psuedorandom procedure to define augmentation. Given a base chance returns True or False. If False,
        chance of the next run of being True increases.
        :return:
        """
        rand = np.random.random()
        if rand < self.run_chance:
            self.run_chance = self.base_chance
            return True
        else:
            self.run_chance += self.step
            if self.run_chance > 1.0:
                self.run_chance = 1.0
            return False

    @staticmethod
    def gen_random_int(_min, _max):
        """
        Returns a random int in given range
        :param _min:
        :param _max:
        :return:
        """
        return np.random.randint(_min, _max)
