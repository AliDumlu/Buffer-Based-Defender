import numpy as np
from scipy.spatial.distance import euclidean, cityblock
from statistics import mean


class ThresholdCalculator:
    def __init__(self, x_org, x_adv):
        if x_org is not None and x_adv is not None:
            self.x_org = x_org
            self.x_adv = x_adv

    def calculate_threshold(self, norm=1):
        distances = []
        if norm == 1:
            norm_func = lambda x, y: euclidean(x, y)
        elif norm == 2:
            norm_func = lambda x, y: cityblock(x, y)
        else:
            norm_func = lambda x, y: np.linalg.norm(x - y, np.inf)
        # Calculate the distances between the original images and the adversarial examples
        for i in range(0, len(self.x_org)):
            distance = norm_func(self.x_org[i].flatten(), self.x_adv[i].flatten())
            distances.append(distance)
        return max(distances), mean(distances)
