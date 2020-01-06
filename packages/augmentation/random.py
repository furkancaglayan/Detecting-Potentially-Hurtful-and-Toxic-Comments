import numpy as np


class RandomMachine(object):
    def __init__(self, base_chance=0.5, step=0.1,random_state=20):
        self.base_chance = base_chance
        self.run_chance = base_chance
        self.step = step

        np.random.seed(random_state)

    def pass_chance(self):
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
        return np.random.randint(_min,_max)
