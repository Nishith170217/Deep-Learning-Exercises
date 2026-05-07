import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size  # [height, width, channels]
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        # Load labels
        with open(self.label_path, 'r') as f:
            self.labels_dict = json.load(f)

        # Build ordered list of image indices
        self.image_ids = sorted(self.labels_dict.keys(), key=lambda x: int(x))
        self.n = len(self.image_ids)

        # Index order (shuffled or not)
        self.indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indices)

        self._current_pos = 0
        self._epoch = 0

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        images = []
        labels = []

        for _ in range(self.batch_size):
            # If we've gone through all data, start a new epoch
            if self._current_pos >= self.n:
                self._current_pos = 0
                self._epoch += 1
                if self.shuffle:
                    np.random.shuffle(self.indices)

            idx = self.indices[self._current_pos]
            img_id = self.image_ids[idx]

            img = np.load(os.path.join(self.file_path, img_id + '.npy'))
            label = int(self.labels_dict[img_id])

            # Resize if needed
            h, w, c = self.image_size
            if img.shape != (h, w, c):
                img = resize(img, (h, w, c), anti_aliasing=True)

            img = self.augment(img)

            images.append(img)
            labels.append(label)

            self._current_pos += 1

        return np.array(images), np.array(labels, dtype=int)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if self.mirroring:
            choice = np.random.randint(0, 3)
            if choice == 0:
                img = np.fliplr(img)  # horizontal flip
            elif choice == 1:
                img = np.flipud(img)  # vertical flip
            # choice == 2: no flip

        if self.rotation:
            k = np.random.randint(1, 4)  # 1, 2, or 3 -> 90, 180, 270 degrees
            img = np.rot90(img, k)

        return img


    def current_epoch(self):
        # return the current epoch number
        return self._epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        n = len(images)
        cols = 3
        rows = int(np.ceil(n / cols))
        fig = plt.figure(figsize=(cols * 3, rows * 3))
        for i in range(n):
            ax = fig.add_subplot(rows, cols, i + 1)
            img = images[i]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            ax.imshow(img)
            ax.set_title(self.class_name(labels[i]))
            ax.axis('off')
        plt.tight_layout()
        plt.show()

