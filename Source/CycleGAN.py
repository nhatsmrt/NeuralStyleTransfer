'''
Note that the img is processed in BGR space (same for vgg net)
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io
import cv2


class CycleGAN():
    def __init__(self, keep_prob=0.9, X_channel = 3, y_channel = 3, is_training=True, cyc_weight = 10):

        self._X_channel = X_channel
        self._y_channel = y_channel

        if is_training:
            self._keep_prob = keep_prob
            self._cyc_weight = cyc_weight


        with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
            self.build(is_training)

    def build(self, is_training):
        self._sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        CONTENT_LAYERS = ['conv5_2']

        # Build generators and discriminators:
        self._is_training = tf.placeholder(tf.bool)
        self._keep_prob_tensor = tf.placeholder(tf.float32)

        self._X = tf.placeholder(shape = [None, None, None, self._X_channel], dtype = tf.float32) # True X
        self._y = tf.placeholder(shape = [None, None, None, self._y_channel], dtype = tf.float32) # True y
        self._y_generated = self.generator(
            self._X,
            op_size = (tf.shape(self._X)[1], tf.shape(self._X)[2]),
            op_channel = self._y_channel,
            name = "gen_X_to_y"
        ) # Generated y from X
        self._X_generated = self.generator(
            self._y,
            op_size = (tf.shape(self._y)[1], tf.shape(self._y)[2]),
            op_channel = self._X_channel,
            name = "gen_y_to_X"
        ) # Generated X from y


        save_list = tf.trainable_variables()
        self._saver = tf.train.Saver(save_list)

        if is_training:
            self._optimizer = tf.train.AdamOptimizer(1e-3)

            self._X_discriminator = self.discriminator(self._X, name="disc_X")
            self._X_generated_discriminator = self.discriminator(self._X_generated, name = "disc_X")
            self._y_discriminator = self.discriminator(self._y, name="disc_y")
            self._y_generated_discriminator = self.discriminator(self._y_generated, name="disc_y")


            self._cyc_X = self.generator(
                self._y_generated,
                op_size= (tf.shape(self._X)[1], tf.shape(self._X)[2]),
                op_channel = self._X_channel,
                name = "gen_y_to_X"
            )
            self._cyc_y = self.generator(
                self._X_generated,
                op_size  = (tf.shape(self._y)[1], tf.shape(self._y)[2]),
                op_channel = self._y_channel,
                name = "gen_X_to_y"
            )

            # Discriminator losses:
            self._d_X_loss_1 = self.mean_square(self._X_discriminator - 1)
            self._d_X_loss_2 = self.mean_square(self._X_generated_discriminator)
            self._d_X_loss = (self._d_X_loss_1 + self._d_X_loss_2) / 2

            self._d_y_loss_1 = self.mean_square(self._y_discriminator - 1)
            self._d_y_loss_2 = self.mean_square(self._y_generated_discriminator)
            self._d_y_loss = (self._d_y_loss_1 + self._d_y_loss_2) / 2

            self._d_loss = self._d_X_loss + self._d_y_loss

            # Generator losses:
            self._g_X_loss = self.mean_square(self._X_generated_discriminator - 1)
            self._g_y_loss = self.mean_square(self._y_generated_discriminator - 1)

            # Cyclic losses:
            self._cyc_loss = self.mean_square(self._X - self._cyc_X) + self.mean_square(self._y - self._cyc_y)
            self._g_loss = self._g_X_loss + self._g_y_loss + self._cyc_weight * self._cyc_loss

            self._total_loss = self._g_loss + self._d_loss

            # Train steps:
            trainable_vars = tf.trainable_variables()
            self._d_vars = [var for var in trainable_vars if 'disc' in var.name]
            self._g_vars = [var for var in trainable_vars if 'gen' in var.name]

            # self._d_X_train_step = self._optimizer.minimize(self._d_X_loss, var_list = self._d_vars)
            # self._d_y_train_step = self._optimizer.minimize(self._d_y_loss, var_list = self._d_vars)
            # self._g_X_train_step = self._optimizer.minimize(self._g_X_loss + 10 * self._cyc_loss, var_list = self._g_vars)
            # self._g_y_train_step = self._optimizer.minimize(self._g_y_loss + 10 * self._cyc_loss, var_list = self._g_vars)

            self._d_train_step = self._optimizer.minimize(self._d_loss, var_list = self._d_vars)
            self._g_train_step = self._optimizer.minimize(self._g_loss, var_list = self._g_vars)


            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                # self._train_step = [self._d_X_train_step, self._d_y_train_step, self._g_X_train_step, self._g_y_train_step]
                self._train_step = [self._d_train_step, self._g_train_step]

        self._init_op = tf.global_variables_initializer()
        self._sess = tf.Session()
        # self._saver = tf.train.Saver()

    def generator(self, x, name, op_size, op_channel):
        x_norm = tf.layers.batch_normalization(x, training=self._is_training)
        conv_layer_1 = self.convolutional_module(
            x = x_norm,
            inp_channel=3,
            op_channel=4,
            name = name + "module_1",
        )

        res_1 = self.residual_module(conv_layer_1, name = name + "res_1", inp_channel=4)
        res_2 = self.residual_module(res_1, name = name + "res_2", inp_channel=4)
        res_3 = self.residual_module(res_2, name = name + "res_3", inp_channel=4)
        res_4 = self.residual_module(res_3, name = name + "res_4", inp_channel=4)

        # self._deconv_1 = tf.nn.sigmoid(self.deconvolutional_layer(
        #     self._res_4,
        #     inp_shape = [self._batch_size, self._img_h / 2, self._img_w / 2, 4],
        #     op_shape = [self._batch_size, self._img_h, self._img_h, 3],
        #     kernel_size = 3,
        #     strides = 2,
        #     padding = 'SAME',
        #     name = "deconv1",
        #     activated = False
        # ))
        resized = self.resize_convolution_layer(
            res_4,
            new_h = op_size[0],
            new_w = op_size[1],
            inp_channel=4,
            op_channel = op_channel,
            name = name + "resized"
        )
        op = resized * 255
        return op

    def discriminator(self, x, name):
        x_h = tf.shape(x)[1]
        x_w = tf.shape(x)[2]
        x_channel = tf.shape(x)[3]

        x_norm = tf.layers.batch_normalization(x, training=self._is_training)

        conv_layer_1 = self.convolutional_module(
            x = x_norm,
            inp_channel = 3,
            op_channel = 4,
            name = name + "module_1",
        )

        res_1 = self.residual_module(conv_layer_1, name = name + "res_1", inp_channel=4)
        res_2 = self.residual_module(res_1, name = name + "res_2", inp_channel=4)
        res_3 = self.residual_module(res_2, name = name + "res_3", inp_channel=4)
        res_4 = self.residual_module(res_3, name = name + "res_4", inp_channel=4)

        res_4_pooled = self.global_average_pooling(res_4)
        fc = self.feed_forward(
            res_4_pooled,
            inp_channel = 4,
            op_channel = 1, op_layer = True,
            name = name + "_fc"
        )
        op = tf.nn.sigmoid(fc)

        return op

    # Define layers and modules:
    def convolutional_layer(self, x, name, inp_channel, op_channel, kernel_size=3, strides=1, padding='VALID',
                            pad=1, dropout=False, not_activated=False, not_normed=False):
        if pad != 0:
            x_padded = tf.pad(x, self.create_pad(4, pad))
        else:
            x_padded = x
        W_conv = tf.get_variable("W_" + name, shape=[kernel_size, kernel_size, inp_channel, op_channel],
                                 initializer=tf.keras.initializers.he_normal())
        b_conv = tf.get_variable("b_" + name, initializer=tf.zeros(op_channel))
        z_conv = tf.nn.conv2d(x_padded, W_conv, strides=[1, strides, strides, 1], padding=padding) + b_conv
        a_conv = tf.nn.relu(z_conv)
        if dropout:
            a_conv_dropout = tf.nn.dropout(a_conv, keep_prob=self._keep_prob)
            return a_conv_dropout
        if not_activated:
            return z_conv
        if not_normed:
            return a_conv

        h_conv = tf.layers.batch_normalization(a_conv, training = self._is_training, renorm = True)
        return h_conv

    def depthwise_separable_conv_layer(self, x, name, inp_channel, op_channel, depth_kernel,
                                       depth_multiplier=1, strides=1, pad=0, padding='SAME'):
        if pad != 0:
            x_padded = tf.pad(x, self.create_pad(4, pad))
        else:
            x_padded = x

        depth_filter = tf.get_variable(
            name=name + "_depth_kernel",
            shape=[depth_kernel, depth_kernel, inp_channel, depth_multiplier]
        )
        point_filter = tf.get_variable(
            name=name + "point_filter",
            shape=[1, 1, inp_channel * depth_multiplier, op_channel]
        )
        separable_conv = tf.nn.separable_conv2d(
            input=x,
            depthwise_filter=depth_filter,
            pointwise_filter=point_filter,
            strides=[1, strides, strides, 1],
            padding=padding,
            name=name + "_separable_conv"
        )
        return separable_conv

    def deconvolutional_layer(self, x, name, inp_shape, op_shape, kernel_size=3, strides=1, padding='VALID',
                              activated=False):
        b_deconv = tf.get_variable("b" + name, initializer=tf.zeros(op_shape[3]))
        filter = tf.get_variable("filter" + name, shape=[kernel_size, kernel_size, op_shape[3], inp_shape[3]])
        z_deconv = tf.nn.conv2d_transpose(x, filter=filter, strides=[1, strides, strides, 1], padding=padding,
                                          output_shape=tf.stack(op_shape)) + b_deconv
        if activated:
            a_deconv = tf.nn.relu(z_deconv)
            h_deconv = tf.layers.batch_normalization(a_deconv, training = self._is_training)
            return h_deconv

        return z_deconv

    def resize_convolution_layer(self, x, name, new_h, new_w, inp_channel, op_channel):
        x_resized = tf.image.resize_images(x, size=(new_h, new_w))
        x_resized_conv = tf.nn.sigmoid(self.convolutional_layer(
            x_resized,
            inp_channel=inp_channel,
            op_channel=op_channel,
            name=name + "_conv",
            not_activated=True
        ))
        return x_resized_conv

    def convolutional_module(self, x, inp_channel, op_channel, name):
        conv1 = self.convolutional_layer(x, inp_channel=inp_channel, op_channel=op_channel, name=name + "_conv1")
        conv2 = self.convolutional_layer(conv1, inp_channel=op_channel, op_channel=op_channel, name=name + "_conv2",
                                         strides=2)
        return conv2

    def convolutional_module_with_max_pool(self, x, inp_channel, op_channel, name, strides=1):
        conv1 = self.convolutional_layer(x, inp_channel=inp_channel, op_channel=op_channel, name=name + "_conv1",
                                         strides=strides)
        conv2 = self.convolutional_layer(conv1, inp_channel=op_channel, op_channel=op_channel, name=name + "_conv2",
                                         strides=strides)
        conv2_max_pool = self.max_pool_2x2(conv2)

        return conv2_max_pool

    def convolution_module_with_more_max_pool(self, x, inp_channel, op_channel, name):
        conv1 = self.convolutional_layer(x, inp_channel=inp_channel, op_channel=op_channel, name=name + "_conv1")
        conv1_max_pool = self.max_pool_2x2(conv1)
        conv2 = self.convolutional_layer(conv1_max_pool, inp_channel=op_channel, op_channel=op_channel,
                                         name=name + "_conv2")
        conv2_max_pool = self.max_pool_2x2(conv2)

        return conv2_max_pool

    def residual_module(self, x, name, inp_channel):
        conv1 = self.convolutional_layer(x, name + "_conv1", inp_channel, inp_channel)
        conv2 = self.convolutional_layer(conv1, name + "_conv2", inp_channel, inp_channel, not_activated=True)
        res_layer = tf.nn.relu(tf.add(conv2, x, name="res"))

        batch_norm = tf.layers.batch_normalization(res_layer, training=self._is_training, renorm = True)

        return batch_norm

    def inception_module(self, x, name, inp_channel, op_channel):
        tower1_conv1 = self.convolutional_layer(x, kernel_size=1, padding='SAME', inp_channel=inp_channel,
                                                op_channel=op_channel // 3, name=name + "_tower1_conv1", pad=0)
        tower1_conv2 = self.convolutional_layer(tower1_conv1, kernel_size=3, padding='SAME',
                                                inp_channel=op_channel // 3, op_channel=op_channel // 3,
                                                name=name + "_tower1_conv2", pad=0)

        tower2_conv1 = self.convolutional_layer(x, kernel_size=1, padding='SAME', inp_channel=inp_channel,
                                                op_channel=op_channel // 3, name=name + "_tower2_conv1", pad=0)
        tower2_conv2 = self.convolutional_layer(tower2_conv1, kernel_size=5, padding='SAME',
                                                inp_channel=op_channel // 3, op_channel=op_channel // 3,
                                                name=name + "_tower2_conv2", pad=0)

        tower3_max_pool = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        tower3_conv = self.convolutional_layer(tower3_max_pool, name=name + "_tower3_conv", inp_channel=inp_channel,
                                               op_channel=op_channel // 3, kernel_size=1, pad=0)

        return tf.concat([tower1_conv2, tower2_conv2, tower3_conv], axis=-1)

    def xception_module(self, x, name, inp_channel):
        # x_activated = tf.nn.relu(x)
        sep_conv_1 = self.depthwise_separable_conv_layer(
            x,
            name=name + "sep_conv_1",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel=inp_channel
        )
        sep_conv_1_norm = tf.layers.batch_normalization(sep_conv_1, training=self._is_training)
        sep_conv_1_norm_activated = tf.nn.relu(sep_conv_1_norm)

        sep_conv_2 = self.depthwise_separable_conv_layer(
            sep_conv_1_norm_activated,
            name=name + "sep_conv_2",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel=inp_channel
        )
        sep_conv_2_norm = tf.layers.batch_normalization(sep_conv_2, training=self._is_training)
        sep_conv_2_norm_activated = tf.nn.relu(sep_conv_2_norm)

        sep_conv_3 = self.depthwise_separable_conv_layer(
            sep_conv_2_norm_activated,
            name=name + "sep_conv_3",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel=inp_channel
        )
        sep_conv_3_norm = tf.layers.batch_normalization(sep_conv_3, training=self._is_training, renorm=True)

        res = tf.nn.relu(x + sep_conv_3_norm)
        return tf.layers.batch_normalization(res, training=self._is_training)

    def squeeze(self, x):
        return self.global_average_pooling(x)

    def excite(self, x, name, n_channels, reduction_ratio=16):
        x_shape = tf.shape(x)
        W_1 = tf.get_variable(shape=[n_channels, n_channels // reduction_ratio], name=name + "_W1")
        z_1 = tf.nn.relu(tf.matmul(x, W_1))
        W_2 = tf.get_variable(shape=[n_channels // reduction_ratio, n_channels], name=name + "_W2")
        return tf.nn.sigmoid(tf.matmul(z_1, W_2))

    def se_block(self, x, name, n_channels):
        x_shape = tf.shape(x)
        x_squeezed = self.squeeze(x)
        x_excited = self.excite(x_squeezed, name=name + "_excited", n_channels=n_channels)
        x_excited_broadcasted = tf.reshape(x_excited, shape=[x_shape[0], 1, 1, x_shape[-1]])
        return tf.multiply(x, x_excited_broadcasted)

    def residual_module_with_se(self, x, name, inp_channel):
        conv1 = self.convolutional_layer(x, name + "_conv1", inp_channel, inp_channel)
        conv2 = self.convolutional_layer(conv1, name + "_conv2", inp_channel, inp_channel, not_activated=True)
        conv2_se = self.se_block(conv2, name=name + "_se", n_channels=inp_channel)
        res_layer = tf.nn.relu(tf.add(conv2_se, x, name=name + "res"))
        batch_norm = tf.layers.batch_normalization(res_layer, training=self._is_training, renorm=True)
        return batch_norm

    def xception_module_with_se(self, x, name, inp_channel):
        # x_activated = tf.nn.relu(x)
        sep_conv_1 = self.depthwise_separable_conv_layer(
            x,
            name=name + "sep_conv_1",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel=inp_channel
        )
        sep_conv_1_norm = tf.layers.batch_normalization(sep_conv_1, training=self._is_training)
        sep_conv_1_norm_activated = tf.nn.relu(sep_conv_1_norm)

        sep_conv_2 = self.depthwise_separable_conv_layer(
            sep_conv_1_norm_activated,
            name=name + "sep_conv_2",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel=inp_channel
        )
        sep_conv_2_norm = tf.layers.batch_normalization(sep_conv_2, training=self._is_training)
        sep_conv_2_norm_activated = tf.nn.relu(sep_conv_2_norm)

        sep_conv_3 = self.depthwise_separable_conv_layer(
            sep_conv_2_norm_activated,
            name=name + "sep_conv_3",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel=inp_channel
        )
        sep_conv_3_norm = tf.layers.batch_normalization(sep_conv_3, training=self._is_training)
        sep_conv_3_norm_se = self.se_block(sep_conv_3_norm, name=name + "_se", n_channels=inp_channel)

        res = tf.nn.relu(x + sep_conv_3_norm_se)
        batch_norm = tf.layers.batch_normalization(res, training=self._is_training, renorm=True)
        return batch_norm

    def hypercolumn(self, layers_list, input_dim):
        layers_list_upsampled = []
        for layer in layers_list:
            layers_list_upsampled.append(tf.image.resize_bilinear(images=layer, size=(input_dim, input_dim)))
        return tf.concat(layers_list, axis=0)

    def feed_forward(self, x, name, inp_channel, op_channel, op_layer=False):
        W = tf.get_variable("W_" + name, shape=[inp_channel, op_channel], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b_" + name, shape=[op_channel], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        z = tf.matmul(x, W) + b
        if op_layer:
            # a = tf.nn.sigmoid(z)
            # return a
            return z
        else:
            a = tf.nn.relu(z)
            a_norm = tf.layers.batch_normalization(a, training=self._is_training)
            return a_norm

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def global_average_pooling(self, x):
        return tf.reduce_mean(x, axis=[1, 2])

    # Predict:
    def generate(self, X, batch_size=None, mode = 'X_to_y'):
        if mode == 'X_to_y':
            generator = self._y_generated
            placeholder = self._X
        else:
            generator = self._X_generated
            placeholder = self._y

        if batch_size is None:
            batch_size = X.shape[0]

        if len(X.shape) == 4:
            train_indicies = np.arange(X.shape[0])
            predictions = np.zeros(shape=X.shape)

            for i in range(int(math.ceil(X.shape[0] / batch_size))):
                # generate indicies for the batch
                start_idx = (i * batch_size) % X.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                # Generate images for this batch
                feed_dict = {
                    placeholder: X[idx, :],
                    self._is_training: False,
                    self._keep_prob_tensor: 1.0
                }
                op = self._sess.run(generator, feed_dict = feed_dict)

                predictions[idx, :, :, :] = op

            return np.round(predictions).astype(np.uint8)
        else:
            predictions = []

            for img in X:
                img = np.array([img])

                feed_dict = {
                    placeholder: img,
                    self._is_training: False,
                    self._keep_prob_tensor: 1.0
                }
                prediction = self._sess.run(generator, feed_dict=feed_dict)

                predictions.append(np.round(prediction[0]).astype(np.uint8))

            return np.array(predictions)

    # Train:
    def train(self, X, y, X_val = None, y_val = None, num_epoch=1, batch_size=16, patience=None, weight_save_path=None,
            weight_load_path=None,
            plot_losses=False, draw_img=True, print_every=1):

        self._sess.run(self._init_op)
        if weight_load_path is not None:
            self._saver.restore(sess=self._sess, save_path=weight_load_path)
            print("Weight loaded successfully")


        if num_epoch > 0:
            print('Training CycleGAN for ' + str(num_epoch) + ' epochs')
            X_indicies = np.arange(X.shape[0])
            y_indicies = np.arange(y.shape[0])

            val_losses = []
            early_stopping_cnt = 0
            it_cnt = 0

            n_batch = min(X.shape[0], y.shape[0]) // batch_size

            for e in range(num_epoch):
                print("Epoch " + str(e + 1))
                np.random.shuffle(X_indicies)
                np.random.shuffle(y_indicies)

                for i in range(n_batch):
                    start_idx = batch_size * i
                    X_idx = X_indicies[start_idx: start_idx + batch_size]
                    y_idx = y_indicies[start_idx: start_idx + batch_size]

                    # Train networks:
                    feed_dict = {
                        self._X: X[X_idx, :],
                        self._y: y[y_idx, :],
                        self._is_training: True,
                        self._keep_prob_tensor: self._keep_prob
                    }
                    loss = self._sess.run(self._total_loss, feed_dict = feed_dict)
                    self._sess.run(self._g_train_step, feed_dict = feed_dict)
                    self._sess.run(self._d_train_step, feed_dict = feed_dict)

                    if it_cnt % print_every == 0:
                        print("Iteration " + str(i) + " with loss " + str(loss))

                    it_cnt += 1

                if X_val is not None:
                    feed_dict = {
                        self._X: X_val,
                        self._y: y_val,
                        self._is_training: False,
                        self._keep_prob_tensor: 1.0
                    }

                    val_loss = self._sess.run(self._total_loss, feed_dict = feed_dict)
                    val_losses.append(val_loss)

                    if e % print_every == 0:
                        print("Validation Total Loss: " + str(val_loss))

                    if val_loss <= min(val_losses) and weight_save_path is not None:
                        save_path = self._saver.save(self._sess, save_path=weight_save_path)
                        print("Model's weights saved at %s" % save_path)

                    if patience is not None:
                        if val_loss > min(val_losses):
                            early_stopping_cnt += 1
                        else:
                            early_stopping_cnt = 0

                        if early_stopping_cnt > patience:
                            print("Patience exceeded. Finish training")
                            return

                    if draw_img:
                        fig = plt.figure()
                        a = fig.add_subplot(2, 2, 1)
                        plt.imshow(cv2.cvtColor(X_val[0], cv2.COLOR_BGR2RGB))

                        a = fig.add_subplot(2, 2, 2)
                        prediction = self.generate(X_val[:1, :], mode = 'X_to_y')[0]
                        plt.imshow(cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB))

                        a = fig.add_subplot(2, 2, 3)
                        plt.imshow(cv2.cvtColor(y_val[0], cv2.COLOR_BGR2RGB))

                        a = fig.add_subplot(2, 2, 4)
                        prediction_2 = self.generate(y_val[:1, :], mode = 'y_to_X')[0]
                        plt.imshow(cv2.cvtColor(prediction_2, cv2.COLOR_BGR2RGB))

                        plt.show()

    def mean_square(self, x):
        return tf.reduce_mean(tf.square(x))

    def create_pad(self, n, pad):
        pad_matrix = [[0, 0]]
        for i in range(n - 2):
            pad_matrix.append([pad, pad])
        pad_matrix.append([0, 0])
        return tf.constant(pad_matrix)

    def save_weights(self, weight_save_path):
        self._saver.save(sess=self._sess, save_path=weight_save_path)
        print("Weights saved successfully")

    def load_weights(self, weight_load_path):
        self._sess.run(self._init_op)
        self._saver.restore(self._sess, weight_load_path)
        print("Weights loaded successfully")

