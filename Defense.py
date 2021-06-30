import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
from minisom import MiniSom


class Defense:

    def __init__(self, n: int, threshold, model=None, norm=1, som_enabled=False, som_data=None):
        self.n = n
        self.model = model
        if norm == 1:
            self.norm = lambda x, y: euclidean(x, y)
        elif norm == 2:
            self.norm = lambda x, y: cityblock(x, y)
        else:
            self.norm = lambda x, y: np.linalg.norm(x - y, np.inf)
        self.threshold = threshold
        self.som_enabled = som_enabled
        if som_enabled:
            self.som = MiniSom(5, 5, som_data.shape[1], sigma=0.75, learning_rate=0.1)
            self.som.train(som_data, 3500, random_order=True)
            self.buffer = [[[] for _ in range(5)] for _ in range(5)]
        else:
            self.buffer = []

    def predict(self, x):
        if self.detect(x):
            return self.model.predict(x)
        else:
            print("Adversarial example has been detected.")
            return None

    def detect(self, x):
        x = x.flatten()
        if self.som_enabled:
            bmu = self.som.winner(x)
            cluster = self.buffer[bmu[0]][bmu[1]]
            for i in range(0, len(cluster)):
                distance = self.norm(x, cluster[i])
                if distance <= self.threshold:
                    return True
            if len(cluster) == self.n:
                cluster.pop(0)
            cluster.append(x)
            return False
        else:
            for i in range(0, len(self.buffer)):
                distance = self.norm(x, self.buffer[i])
                if distance <= self.threshold:
                    return True
            if len(self.buffer) == self.n:
                self.buffer.pop(0)
            self.buffer.append(x)
            return False
