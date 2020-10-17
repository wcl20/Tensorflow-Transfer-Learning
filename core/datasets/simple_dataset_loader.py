import cv2
import numpy as np
import os
import tqdm

class SimpleDatasetLoader:

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

    def load(self, img_paths):

        data = []; labels = []
        for img_path in tqdm.tqdm(img_paths):
            image = cv2.imread(img_path)
            label = img_path.split(os.path.sep)[-2]

            if self.preprocessors:
                for preprocessor in self.preprocessors:
                    image = preprocessor.preprocess(image)

            data.append(image)
            labels.append(label)

        return np.array(data), np.array(labels)
