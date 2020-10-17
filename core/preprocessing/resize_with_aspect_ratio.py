import cv2
import imutils

class ResizeWithAspectRatio:

    def __init__(self, height, width, interpolation=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def preprocess(self, image):
        img_height, img_width = image.shape[:2]

        # Resize along smaller dimension
        if img_width < img_height:
            image = imutils.resize(image, width=self.width, inter=self.interpolation)
            # Compute change in height + center crop
            offset = int((image.shape[0] - self.height) / 2)
            image = image[offset:offset+self.height, :]
        else:
            image = imutils.resize(image, height=self.height, inter=self.interpolation)
            # Compute change in width + center crop
            offset = int((image.shape[1] - self.width) / 2)
            image = image[:, offset:offset+self.width]

        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)
