import numpy as np
import load_data
import os
from keras.layers import Input, Dropout, Cropping2D, Concatenate, Dense, Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.losses import mse

def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session())

class Trainer:
    def __init__(self, data_path, model_path=os.path.abspath(__file__)[0:-11], d_learning_rate=0.00002,
                 g_learning_rate=0.00002, metric_weight=1, adversarial_weight=50):
        self.img_shape = (64, 64, 1)
        self.data_path = data_path
        self.model_path = model_path
        self.data_loader = load_data.data_reader(data_path)
        self.data_loader.create_training_data()
        self.norm_size = self.data_loader.norm_size
        self.generator = self.build_generator()
        # self.generator.summary()
        self.discriminator = self.build_discriminator()
        # self.discriminator.summary()
        self.discriminator.compile(loss=mse, optimizer=Adam(d_learning_rate, beta_1=0.5), metrics=['accuracy'])

        ## building of combain model
        # img_a: input image img_b: conditioning image
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        fake_A= self.generator(img_B)

        self.discriminator.trainable = False

        valid = self.discriminator([fake_A, img_B])
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[adversarial_weight, metric_weight],
                              optimizer=Adam(g_learning_rate, beta_1=0.5))

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
        d1 = Conv2D(filters=100, kernel_size=4, strides=2, padding='same')(d0)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1 = Dropout(0.35)(d1)
        d2 = Conv2D(filters=200, kernel_size=4, strides=2, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization(momentum=0.8)(d2)
        d2 = Dropout(0.4)(d2)
        d3 = Conv2D(filters=400, kernel_size=4, strides=2, padding='same')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = BatchNormalization(momentum=0.8)(d3)
        d3 = Dropout(0.45)(d3)
        d4 = Conv2D(filters=800, kernel_size=4, strides=2, padding='same')(d3)
        d4 = LeakyReLU(alpha=0.2)(d4)
        d4 = BatchNormalization(momentum=0.8)(d4)
        d4 = Dropout(0.5)(d4)
        d5 = Conv2D(filters=1600, kernel_size=4, strides=2, padding='same')(d4)
        d5 = LeakyReLU(alpha=0.2)(d5)
        d5 = BatchNormalization(momentum=0.8)(d5)
        d5 = Dropout(0.5)(d5)
        d6 = Conv2D(filters=1600, kernel_size=4, strides=2, padding='same')(d5)
        d6 = LeakyReLU(alpha=0.2)(d6)
        d6 = BatchNormalization(momentum=0.8)(d6)
        d6 = Dropout(0.5)(d6)
        # up sampling
        u6 = deconve(d6, d5, 800, 0.5)
        u5 = deconve(u6, d4, 800, 0.5)
        u4 = deconve(u5, d3, 400, 0.45)
        u3 = deconve(u4, d2, 200, 0.4)
        u2 = deconve(u3, d1, 100, 0.35)
        u1 = UpSampling2D(size=2)(u2)
        output_layer = Conv2D(filters=1, kernel_size=4, strides=1, padding='same', activation='tanh')(u1)

        return Model(inputs=[d0], outputs=[output_layer])

    def build_discriminator(self):
        img_A=Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        dis_Input=Concatenate(axis=-1)([img_A, img_B])
        d1 = Conv2D(filters=4, kernel_size=4, strides=2, padding='same')(dis_Input)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1 = Dropout(0.3)(d1)
        d2 = Conv2D(filters=8, kernel_size=4, strides=4, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization(momentum=0.2)(d2)
        d2 = Dropout(0.4)(d2)
        d3 = Conv2D(filters=10, kernel_size=4, strides=4, padding='same')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = BatchNormalization(momentum=0.8)(d3)
        d3 = Dropout(0.5)(d3)
        d3_f = Flatten()(d3)
        valid = Dense(1, activation='tanh')(d3_f)
        return Model([img_A, img_B], valid)

    def train(self, epochs=1, batch_size=1, is_retrain=0, g_w="", d_w=""):
        delay = 0
        T_label = np.ones(batch_size, )
        F_label = -np.ones(batch_size, )
        if is_retrain:
            self.discriminator.load_weights(os.path.join(self.model_path, d_w))
            self.generator.load_weights(os.path.join(self.model_path, g_w))
        for epoch in range(epochs):
            imgs_A, imgs_B = self.data_loader.load_epoch_layer1()
            n_batch = int(imgs_A.shape[0]/batch_size)
            for batch_i in range(n_batch):
                imgs_A_batch = imgs_A[batch_i*batch_size:(batch_i+1)*batch_size, :, :]/1100
                imgs_B_batch = imgs_B[batch_i*batch_size:(batch_i+1)*batch_size, :, :]/1100
                imgs_A_batch = np.expand_dims(imgs_A_batch, axis=-1)
                imgs_B_batch = np.expand_dims(imgs_B_batch, axis=-1)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                if delay < 1:
                    fake_A = self.generator.predict([imgs_B_batch])
                    d_loss_real = self.discriminator.train_on_batch([imgs_A_batch, imgs_B_batch], T_label)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B_batch], F_label)
                    d_loss = 0.5*np.add(d_loss_real, d_loss_fake)
                    if d_loss[1] > 0.88:
                        delay = 200
                else:
                    delay -= 1
                # ---------------------
                #  Train generator
                # ---------------------
                g_losss = self.combined.train_on_batch([imgs_A_batch, imgs_B_batch], [T_label, imgs_A_batch])
                if delay < 1:
                    print("Epoch: %d/%d, Batch: %d/%d, discriminator loss: %1.2f, discriminator acc: %1.2f, generator loss: %1.2f"
                        % (epoch+1, epochs, batch_i+1, n_batch, d_loss[0], d_loss[1], g_losss[0]))
                else:
                    print("Epoch: %d/%d, Batch: %d/%d, discriminator training pause, generator loss: %1.2f"
                        % (epoch+1, epochs, batch_i+1, n_batch, g_losss[0]))
            if epoch % 3 == 0 or epoch == epochs-1:
                print('saving models...')
                self.generator.save(os.path.join(self.model_path, "G_model_layer1.h5"))
                self.discriminator.save(os.path.join(self.model_path, "D_model_layer1.h5"))


