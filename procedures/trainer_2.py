import numpy as np
import matplotlib.pyplot as plt
import load_data
import os
from keras.layers import Input, Dropout, Cropping2D, Concatenate, Dense, Flatten
from keras.layers import BatchNormalization, ZeroPadding2D, Add
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.losses import mse
import cv2

def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session())

class Trainer:
    def __init__(self, data_path, model_path=os.path.abspath(__file__)[0:-12]):
        self.img_shape = (64, 64, 1)
        self.data_path = data_path
        self.model_path = model_path
        self.data_loader = load_data.data_reader(data_path)
        self.data_loader.create_training_data()
        self.norm_size = self.data_loader.norm_size
        self.crop_frame = self.data_loader.crop_frame
        self.generator = self.build_generator()
        # self.generator.summary()
        self.discriminator = self.build_discriminator()
        # self.discriminator.summary()
        self.discriminator.compile(loss=mse, optimizer=Adam(0.00012, beta_1=0.5), metrics=['accuracy'])

        ## building of combain model
        # img_a: input image img_b: conditioning image
        img_D = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        img_C = Input(shape=self.img_shape)

        fake_middle = self.generator(img_D)
        padd_fake = ZeroPadding2D(padding=((self.crop_frame,  self.crop_frame + 1),
                                           (self.crop_frame,  self.crop_frame + 1)))(fake_middle)
        fake = Add()([img_B, padd_fake])
        self.discriminator.trainable = False

        valid = self.discriminator([fake, img_C])
        self.combined = Model(inputs=[img_D, img_C, img_B], outputs=[valid, fake])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[4, 160], optimizer=Adam(0.000008, beta_1=0.5))

    def build_generator(self):
        def deconve(layer_input, skip_input, filters, dropout_rate):
            f_size = 4
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            cw = (u.get_shape()[2]-skip_input.get_shape()[2]).value
            ch = (u.get_shape()[1] - skip_input.get_shape()[1]).value
            crop_u = Cropping2D(cropping=((int(ch/2), int(ch/2) + (ch % 2)), (int(cw/2), int(cw/2) + (cw % 2))))(u)
            u = Concatenate()([crop_u, skip_input])
            return u
        d0 = Input(self.img_shape, name="input_img")
        # down sampling
        d1 = Conv2D(filters=250, kernel_size=4, strides=2, padding='same')(d0)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1 = Dropout(0.2)(d1)
        d1_u = Conv2D(filters=250, kernel_size=4, strides=2, padding='same')(d0)
        d1_u = LeakyReLU(alpha=0.2)(d1_u)
        d1_u = Dropout(0.2)(d1_u)
        d2 = Conv2D(filters=300, kernel_size=4, strides=2, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization(momentum=0.8)(d2)
        d2 = Dropout(0.3)(d2)
        d2_u = Conv2D(filters=300, kernel_size=4, strides=2, padding='same')(d1)
        d2_u = LeakyReLU(alpha=0.2)(d2_u)
        d2_u = BatchNormalization(momentum=0.8)(d2_u)
        d2_u = Dropout(0.3)(d2_u)
        d3 = Conv2D(filters=400, kernel_size=4, strides=2, padding='same')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = BatchNormalization(momentum=0.8)(d3)
        d3 = Dropout(0.35)(d3)
        d4 = Conv2D(filters=800, kernel_size=4, strides=2, padding='same')(d3)
        d4 = LeakyReLU(alpha=0.2)(d4)
        d4 = BatchNormalization(momentum=0.8)(d4)
        d4 = Dropout(0.4)(d4)
        d5 = Conv2D(filters=1600, kernel_size=4, strides=2, padding='same')(d4)
        d5 = LeakyReLU(alpha=0.2)(d5)
        d5 = BatchNormalization(momentum=0.8)(d5)
        d5 = Dropout(0.5)(d5)
        d6 = Conv2D(filters=2000, kernel_size=4, strides=2, padding='same')(d5)
        d6 = LeakyReLU(alpha=0.2)(d6)
        d6 = BatchNormalization(momentum=0.8)(d6)
        d6 = Dropout(0.5)(d6)
        # up sampling
        u6 = deconve(d6, d5, 1000, 0.5)
        u5 = deconve(u6, d4, 800, 0.4)
        u4 = deconve(u5, d3, 400, 0.35)
        u3 = deconve(u4, d2_u, 300, 0.3)
        u2 = deconve(u3, d1_u, 250, 0.3)
        u1 = UpSampling2D(size=2)(u2)
        u0_1 = Conv2D(filters=60, kernel_size=4, strides=1, padding='same', activation='relu')(u1)
        u0_2 = Conv2D(filters=30, kernel_size=4, strides=1, padding='same', activation='relu')(u0_1)
        u0_3 = Conv2D(filters=15, kernel_size=4, strides=1, padding='same', activation='relu')(u0_2)
        output_layer = Conv2D(filters=1, kernel_size=4, strides=1, padding='same', activation='relu')(u0_3)
        crop_output_layer = Cropping2D(cropping=((self.crop_frame, self.crop_frame + 1),
                                                 (self.crop_frame, self.crop_frame + 1)))(output_layer)
        return Model(inputs=[d0], outputs=[crop_output_layer])

    def build_discriminator(self):
        img_A=Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        dis_Input=Concatenate(axis=-1)([img_A, img_B])
        d1 = Conv2D(filters=20, kernel_size=4, strides=2, padding='same')(dis_Input)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1 = Dropout(0.35)(d1)
        d2 = Conv2D(filters=40, kernel_size=4, strides=2, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization(momentum=0.2)(d2)
        d2 = Dropout(0.4)(d2)
        d3 = Conv2D(filters=60, kernel_size=4, strides=2, padding='same')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = BatchNormalization(momentum=0.8)(d3)
        d3 = Dropout(0.4)(d3)
        d4 = Conv2D(filters=100, kernel_size=4, strides=4, padding='same')(d3)
        d4 = LeakyReLU(alpha=0.2)(d4)
        d4 = BatchNormalization(momentum=0.8)(d4)
        d4 = Dropout(0.5)(d4)
        d4_f = Flatten()(d4)
        valid = Dense(1, activation='tanh')(d4_f)
        return Model([img_A, img_B], valid)

    def train(self, first_generator, sigma1,  sigma2, epochs=1, batch_size=1, is_retrain=0, g_w="", d_w=""):
        delay = 0
        if is_retrain:
            self.discriminator.load_weights(os.path.join(self.model_path, d_w))
            self.generator.load_weights(os.path.join(self.model_path, g_w))
        for epoch in range(epochs):
            imgs_A, imgs_B, imgs_C = self.data_loader.load_batch_2(sigma1)
            n_batch = int(imgs_A.shape[0] / batch_size)
            if epoch % 10 == 0 and epoch != 0:
                sigma1 = sigma1*0.996
            for batch_i in range(n_batch):
                imgs_A_batch = imgs_A[batch_i*batch_size:(batch_i+1)*batch_size, :, :]/1100
                imgs_B_batch = imgs_B[batch_i*batch_size:(batch_i+1)*batch_size, :, :]/1100
                imgs_C_batch = np.copy(imgs_B_batch)
                imgs_C_batch[:, self.crop_frame:self.norm_size - self.crop_frame - 1,
                self.crop_frame:self.norm_size - self.crop_frame - 1] = \
                    np.random.randn(batch_size, self.norm_size - 2 * self.crop_frame - 1,
                    self.norm_size - 2 * self.crop_frame - 1) * sigma1 + \
                    imgs_C_batch[:, self.crop_frame:self.norm_size - self.crop_frame - 1,
                    self.crop_frame:self.norm_size - self.crop_frame - 1]
                imgs_A_batch = np.expand_dims(imgs_A_batch, axis=-1)
                imgs_B_batch = np.expand_dims(imgs_B_batch, axis=-1)
                imgs_D_batch = first_generator.predict(imgs_B_batch)
                imgs_C_batch = np.expand_dims(imgs_C_batch, axis=-1)
                T_label = np.ones(batch_size,)
                F_label = -np.ones(batch_size,)
                # ---------------------
                #  Train Discriminator
                # ---------------------
                if delay < 1:
                    fake_middle = self.generator.predict([imgs_D_batch])
                    fake = np.copy(imgs_B_batch)
                    fake[:, self.crop_frame:self.norm_size - self.crop_frame -1,
                        self.crop_frame:self.norm_size - self.crop_frame -1] = fake_middle
                    d_loss_real = self.discriminator.train_on_batch([imgs_A_batch, imgs_C_batch], T_label)
                    d_loss_fake = self.discriminator.train_on_batch([fake, imgs_C_batch], F_label)
                    d_loss = 0.5*np.add(d_loss_real, d_loss_fake)
                    if d_loss[1] > 0.99:
                        delay = 200
                else:
                    delay -= 1
                # ---------------------
                #  Train generator
                # ---------------------
                if epoch == 0 or epoch > 2:
                    g_losss = self.combined.train_on_batch([imgs_D_batch, imgs_C_batch, imgs_B_batch],
                                                           [T_label, imgs_A_batch])
                if delay < 1 and (epoch == 0 or epoch > 2):
                    print("Epoch: %d/%d, Batch: %d/%d, discriminator loss: %1.2f, discriminator acc: %1.2f, generator loss: %1.2f"
                        % (epoch+1, epochs, batch_i+1, n_batch, d_loss[0], d_loss[1], g_losss[0]))
                elif delay >= 1:
                    print("Epoch: %d/%d, Batch: %d/%d, discriminator training pause, generator loss: %1.2f"
                        % (epoch+1, epochs, batch_i+1, n_batch, g_losss[0]))
                elif 0 < epoch < 3:
                    print("Epoch: %d/%d, Batch: %d/%d, discriminator loss: %1.2f, discriminator acc: %1.2f, generator training pause"
                        % (epoch+1, epochs, batch_i+1, n_batch, d_loss[0], d_loss[1]))
            if epoch % 3 == 0 or epoch == epochs-1:
                print('saving models...')
                self.generator.save(os.path.join(self.model_path, "G_model_layer2g2.h5"))
                self.discriminator.save(os.path.join(self.model_path, "D_model_layer2g2.h5"))
            if epoch == 20:
                print('saving models...')
                self.generator.save(os.path.join(self.model_path, "G_model_layer2g2_20.h5"))
            if epoch == 50:
                    print('saving models...')
                    self.generator.save(os.path.join(self.model_path, "G_model_layer2g2_50.h5"))
            if epoch == 100:
                print('saving models...')
                self.generator.save(os.path.join(self.model_path, "G_model_layer2g2_100.h5"))
            if epoch == 100:
                print('saving models...')
                self.generator.save(os.path.join(self.model_path, "G_model_layer2g2_200.h5"))
            if epoch == 400:
                print('saving models...')
                self.generator.save(os.path.join(self.model_path, "G_model_layer2g2_400.h5"))
            if epoch == 800:
                    print('saving models...')
                    self.generator.save(os.path.join(self.model_path, "G_model_layer2g2_800.h5"))
            if epoch == 1400:
                print('saving models...')
                self.generator.save(os.path.join(self.model_path, "G_model_layer2g2_1400.h5"))




path = "D:\\Breast Cancer\\Databases\\crope images for GAN\\U"
dr = load_data.data_reader(path)
dr.create_training_data()
sigma1 = 0.02
sigma2 = 0
is_train = 1
is_healthy = 1
if is_train:

    GAN_trainer = Trainer(data_path=path)
    f_generator = load_model(os.path.join(os.path.abspath(__file__)[0:-12], "G_model_e6.h5"))
    generator = load_model(os.path.join(os.path.abspath(__file__)[0:-12], "G_model_layer2g_800.h5"))
    discriminator = load_model(os.path.join(os.path.abspath(__file__)[0:-12], "D_model_layer2g.h5"))
    generator.save_weights("g_w.h5")
    discriminator.save_weights("d_w.h5")
    GAN_trainer.train(f_generator, sigma1, sigma2, epochs=2000, batch_size=50, is_retrain=1, g_w="g_w.h5", d_w="d_w.h5")
    #GAN_trainer.train(f_generator, sigma1, sigma2, epochs=2000, batch_size=50)


else:
    f_generator = load_model(os.path.join(os.path.abspath(__file__)[0:-12], "G_model_e6.h5"))
    s_generator = load_model(os.path.join(os.path.abspath(__file__)[0:-12], "G_model_layer2b.h5"))
    if not is_healthy:
        imgs_A, imgs_B, imgs_C = dr.load_batch_2(sigma1)
        for i in range(5):

            img = imgs_B[i, :, :]

            plt.imshow(img/1100, cmap="gray")
            plt.show(block=False)

            img2 = imgs_A[i, :, :]
            plt.figure()
            plt.imshow(img2/1100, cmap="gray")

            plt.show(block=False)

            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)
            print(img.shape)
            fake_first_l = f_generator.predict(img/1100)
            c_fake_first_l = np.copy(fake_first_l)

            fake_first_l = np.squeeze(fake_first_l)

            plt.figure()
            plt.imshow(fake_first_l, cmap="gray")
            plt.show(block=False)

            c_fake_first_l[:, dr.crop_frame:dr.norm_size - dr.crop_frame - 1,
                    dr.crop_frame:dr.norm_size - dr.crop_frame - 1] = \
                    np.random.randn(1, dr.norm_size - 2 * dr.crop_frame - 1,
                    dr.norm_size - 2 * dr.crop_frame - 1, 1) * 0.01 + \
                    c_fake_first_l[:, dr.crop_frame:dr.norm_size - dr.crop_frame - 1,
                    dr.crop_frame:dr.norm_size - dr.crop_frame - 1]
            fake_second_l = s_generator.predict(c_fake_first_l)
            fake_second_l = np.squeeze(fake_second_l)
            plt.figure()
            plt.imshow(fake_second_l, cmap="gray")
            plt.show()
    else:
        img = np.load("D:\\Breast Cancer\\Databases\\crope images for GAN\\H\\RR7392_L_MLO_5_2_2.npy")
        img_A = cv2.resize(img, (64, 64))
        img_B = np.copy(img_A)
        img_B[10:64 - 10 - 1, 10:64 - 10 - 1] = 0
        plt.figure()
        plt.imshow(img_A / 1100, cmap="gray")
        plt.show(block=False)
        plt.figure()
        plt.imshow(img_B / 1100, cmap="gray")
        plt.show(block=False)

        img_C = np.expand_dims(img_B, axis=-1)
        img_C = np.expand_dims(img_C, axis=0)
        fake_first_l = f_generator.predict(img_C/1100)
        c_fake_first_l = np.copy(fake_first_l)
        fake_first_l = np.squeeze(fake_first_l)

        plt.figure()
        plt.imshow(fake_first_l, cmap="gray")
        plt.show(block=False)

        c_fake_first_l[:, dr.crop_frame:dr.norm_size - dr.crop_frame - 1,
            dr.crop_frame:dr.norm_size - dr.crop_frame - 1] = \
            np.random.randn(1, dr.norm_size - 2 * dr.crop_frame - 1,
                            dr.norm_size - 2 * dr.crop_frame - 1, 1) * 0.01 + \
            c_fake_first_l[:, dr.crop_frame:dr.norm_size - dr.crop_frame - 1,
            dr.crop_frame:dr.norm_size - dr.crop_frame - 1]
        fake_second_l = s_generator.predict(c_fake_first_l)
        fake_second_l = np.squeeze(fake_second_l)
        plt.figure()
        plt.imshow(fake_second_l, cmap="gray")
        plt.show()
