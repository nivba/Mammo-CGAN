import numpy as np
import os
import cv2

class data_reader:
    def __init__(self, path=None):
        self.path = path
        self.training_data = []
        self.crop_frame = int(10)
        self.norm_size = int(64)

    def create_training_data(self):
        for img in os.listdir(self.path):
            im_path = os.path.join(self.path, img)
            img_array = np.load(im_path)
            new_img = cv2.resize(img_array, (self.norm_size, self.norm_size))
            # new_img2==> scaling augmentation
            new_img2 = cv2.resize(img_array, (self.norm_size+10, self.norm_size+10))
            new_img2 = new_img2[5:self.norm_size+5, 5:self.norm_size+5]
            #
            self.training_data.append(new_img)
            self.training_data.append(new_img2)
        self.training_data = np.array(self.training_data)
        print(self.training_data.shape)

    def load_epoch_layer1(self):
        idx = np.random.permutation(len(self.training_data))
        batch = self.training_data[idx, :, :]
        Full_images = batch
        m_img = np.copy(batch)
        m_img[:, self.crop_frame:self.norm_size - self.crop_frame - 1,
        self.crop_frame:self.norm_size - self.crop_frame - 1] = 0
        Mask_images = m_img
        return Full_images, Mask_images

    def load_epoch_layer2(self, sigma):
        Full_images, Mask_images = self.load_epoch_layer1()
        Noisy_image = np.copy(Full_images)
        Noisy_image[:, self.crop_frame:self.norm_size - self.crop_frame - 1,
            self.crop_frame:self.norm_size - self.crop_frame - 1] = \
            np.random.randn(self.training_data.shape[0],self.norm_size-2*self.crop_frame-1,
            self.norm_size-2*self.crop_frame-1)*sigma +\
            Noisy_image[:, self.crop_frame:self.norm_size - self.crop_frame - 1,
            self.crop_frame:self.norm_size - self.crop_frame - 1] / 1100
        return Full_images, Mask_images, Noisy_image
