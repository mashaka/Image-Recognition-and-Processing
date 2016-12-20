import numpy as np

LETTER_DEPTH = 3
NT_CONST = 10
NT_ORDER = 3
NT_VARIANCE = 2
MEANS_WEIGHT = 0.2
MIN_MAX_WEIGHT = 0.8
WHITE_SHIFT = 2


class ImagePyramidLayer:
        def __init__(self, previous_layer, first_layer_value=False):
            if first_layer_value:
                self.order = 0
                self.mins = previous_layer
                self.maxs = previous_layer
                self.means = previous_layer
                self.shape = (previous_layer.shape[0], previous_layer.shape[1])
                self.variances = np.zeros(self.shape)
                return
            self.order = previous_layer.order + 1
            if previous_layer.shape[0] % 2 == 1 or previous_layer.shape[0] % 2 == 1:
                raise ValaueError('Previous layer should have even height and width')
            self.shape = (previous_layer.shape[0] // 2, previous_layer.shape[1] // 2)
            self.variances = np.zeros(self.shape)
            self.mins = np.zeros(self.shape)
            self.maxs = np.zeros(self.shape)
            self.means = np.zeros(self.shape)
            for y in range(0, previous_layer.shape[0], 2):
                for x in range(0, previous_layer.shape[1], 2):
                    self.mins[y//2][x//2] = min(
                        previous_layer.mins[y][x],
                        previous_layer.mins[y][x+1],
                        previous_layer.mins[y+1][x],
                        previous_layer.mins[y+1][x+1]
                    )
                    self.maxs[y//2][x//2] = max(
                        previous_layer.maxs[y][x],
                        previous_layer.maxs[y][x+1],
                        previous_layer.maxs[y+1][x],
                        previous_layer.maxs[y+1][x+1]
                    ) 
                    self.means[y//2][x//2] = np.mean([
                        previous_layer.means[y][x],
                        previous_layer.means[y][x+1],
                        previous_layer.means[y+1][x],
                        previous_layer.means[y+1][x+1]
                    ])

        def calc_variances(self, previous_layer):
            self.variances = np.zeros(self.shape)
            for y in range(self.shape[0]):
                for x in range(self.shape[1]):
                    self.variances[y][x] += previous_layer.variances[2*y][2*x] + \
                                    previous_layer.variances[2*y][2*x+1] + \
                                    previous_layer.variances[2*y+1][2*x] + \
                                    previous_layer.variances[2*y+1][2*x+1]
                    self.variances[y][x] += previous_layer.means[2*y][2*x] ** 2 + \
                                    previous_layer.means[2*y][2*x+1] ** 2 + \
                                    previous_layer.means[2*y+1][2*x] ** 2 + \
                                    previous_layer.means[2*y+1][2*x+1] ** 2
                    self.variances[y][x] /= 4
                    self.variances[y][x] -= self.means[y][x] ** 2

        def add_border(self):
            if self.shape[0] % 2 == 1:
                self.mins = np.append(self.mins, [self.mins[-1]], axis=0)
                self.maxs = np.append(self.maxs, [self.maxs[-1]], axis=0)
                self.means = np.append(self.means, [self.means[-1]], axis=0)
                self.variances = np.append(self.variances, [self.variances[-1]], axis=0)
                self.shape = (self.shape[0] + 1, self.shape[1])
            if self.shape[1] % 2 == 1:
                self.mins = np.append(self.mins, 
                    np.array([[self.mins[i][-1]] for i in range(self.mins.shape[0])]), axis=1)
                self.maxs = np.append(self.maxs, 
                    np.array([[self.maxs[i][-1]] for i in range(self.maxs.shape[0])]), axis=1)
                self.means = np.append(self.means, 
                    np.array([[self.means[i][-1]] for i in range(self.means.shape[0])]), axis=1)
                self.variances = np.append(self.variances, 
                    np.array([[self.variances[i][-1]] for i in range(self.variances.shape[0])]), axis=1)
                self.shape = (self.shape[0], self.shape[1] + 1)

        def calc_threshold(self, prev_layer):
            self.thresholds = np.zeros(self.shape)
            for y in range(0, self.shape[0], 2):
                for x in range(self.shape[1] - 1):
                    if x % 2 == 0:
                        self.thresholds[y][x] = prev_layer.thresholds[y//2][x//2]
                    else:
                        self.thresholds[y][x] = prev_layer.thresholds[y//2][x//2] * 0.25 + \
                            prev_layer.thresholds[y//2][x//2+1] * 0.75
                self.thresholds[y][-1] = self.thresholds[y][-2]
            for y in range(1, self.shape[0] - 1, 2):
                self.thresholds[y] = self.thresholds[y-1] * 0.75 + self.thresholds[y+1] * 0.25
            self.thresholds[-1] = self.thresholds[-2]

        def get_noise_threshold(self, y, x):
            return NT_CONST + NT_ORDER * self.order + NT_VARIANCE * np.sqrt(self.variances[y][x])
        
        
        def check_threshold(self, prev_layer):
            self.stop_thresholds = np.zeros(self.shape, dtype=bool)
            for y in range(self.shape[0]):
                for x in range(self.shape[1]):
                    if not prev_layer.stop_thresholds[y//2][x//2]:
                        if self.maxs[y][x] - self.mins[y][x] > self.get_noise_threshold(y, x):
                            self.thresholds[y][x] = self.means[y][x] * MEANS_WEIGHT + \
                                (self.maxs[y][x] + self.mins[y][x]) / 2 * MIN_MAX_WEIGHT + WHITE_SHIFT
                        else:
                            self.stop_thresholds[y][x] = True
                    else:
                        self.stop_thresholds[y][x] = True

        def down_thresholds(self, prev_layer):
            self.thresholds = np.zeros(self.shape)
            for y in range(self.shape[0]):
                for x in range(self.shape[1]):
                    self.thresholds[y][x] = prev_layer.thresholds[y//2][x//2] 

class ImagePyramid:
    def __init__(self, img):
        self.img = img
        self.layers = [ImagePyramidLayer(img, first_layer_value=True)]
        while self.layers[-1].shape[0] > 2 and self.layers[-1].shape[1] > 2:
            self.layers[-1].add_border()
            self.layers.append(ImagePyramidLayer(self.layers[-1]))
            self.layers[-1].calc_variances(self.layers[-2])
    
    def calc_thresholds(self):
        self.layers[-1].thresholds = self.layers[-1].means * MEANS_WEIGHT + \
            (self.layers[-1].maxs + self.layers[-1].mins) / 2 * MIN_MAX_WEIGHT
        self.layers[-1].stop_thresholds = np.zeros(self.layers[-1].shape, dtype=bool)
        for i in range(len(self.layers) - 2, LETTER_DEPTH, -1):
            self.layers[i].calc_threshold(self.layers[i+1])
            self.layers[i].check_threshold(self.layers[i+1])
        for i in range(LETTER_DEPTH, -1, -1):
            self.layers[i].calc_threshold(self.layers[i + 1])
    
    def get_processed_img(self):
        bin_image = np.zeros(self.img.shape)
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                if self.layers[0].thresholds[y][x] - WHITE_SHIFT < self.img[y][x]:
                    bin_image[y][x] = 255
        return bin_image