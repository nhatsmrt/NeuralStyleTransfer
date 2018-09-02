import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy

class StyleTransferNet():
    def __init__(self, style_img, img_h, img_w, img_c = 3, keep_prob = 0.9, pretrained_path = None):
        self._style_img = style_img
        self._img_h = img_h
        self._img_w = img_w
        self._img_c = img_c
        self._keep_prob = keep_prob

        self._X_transformed = self.image_transformation_net()
        STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
        CONTENT_LAYER = ['relu4_2']

        self._style_loss = self.losses(
            self._X_transformed,
            self._style_img,
            layers = STYLE_LAYERS,
            loss_type = self.style_loss_v2,
            path = pretrained_path
        )
        self._feat_loss = self.losses(
            self._X_transformed,
            self._style_img,
            layers = CONTENT_LAYER,
            loss_type = self.feat_loss,
            path = pretrained_path
        )
        self._mean_loss = self._style_loss + self._feat_loss

        self._init_op = tf.global_variables_initializer()
        self._saver = tf.train.Saver()
        self._sess = tf.Session()
        self._sess.run(self._init_op)


    def image_transformation_net(self):
        self._X = tf.placeholder(shape = [None, self._img_h, self._img_w, self._img_c])
        self._batch_size = tf.shape(self._X)[0]
        self._is_training = tf.placeholder(tf.bool)
        self._keep_prob_tensor = tf.placeholder(tf.float32)
        self._X_norm = tf.layers.batch_normalization(self._X, training=self._is_training)

        self._conv_module_1 = self.convolutional_module_with_max_pool(x = self._X_norm, inp_channel = 3, op_channel = 4, name = "module_1", strides = 1)

        self._res_1 = self.residual_module(self._conv_module_1, name = "res_1", inp_channel = 4)
        self._res_2 = self.residual_module(self._res_1, name = "res_2", inp_channel = 4)
        self._res_3 = self.residual_module(self._res_2, name = "res_3", inp_channel = 4)
        self._res_4 = self.residual_module(self._res_3, name = "res_4", inp_channel = 4)

        self._deconv_1 = self.deconvolutional_layer(
            self._res_4,
            inp_shape = [self._batch_size, self._img_h / 2, self._img_w / 2, 4],
            op_shape = [self._batch_size, self._img_h, self._img_h, 3],
            kernel_size = 3,
            strides = 2,
            padding = 'SAME',
            name = "deconv1")
        # self._deconv_2 = self.deconvolutional_layer(
        #     self._deconv_1,
        #     inp_shape = [self._batch_size, 32, 32, 2],
        #     op_shape = [self._batch_size, 64, 64, 1],
        #     kernel_size = 3,
        #     strides = 2,
        #     padding = 'SAME',
        #     name = "deconv2")
        # self._X_reconstructed_batch_norm = tf.reshape(self._deconv_1, shape = [-1, self._img_h * self._img_w])

        return self._deconv_1





    # Define layers and modules:
    def convolutional_layer(self, x, name, inp_channel, op_channel, kernel_size=3, strides = 1, padding='VALID',
                            pad = 1, dropout=False, not_activated=False):
        if pad != 0:
            x_padded = tf.pad(x, self.create_pad(4, pad))
        else:
            x_padded = x
        W_conv = tf.get_variable("W_" + name, shape=[kernel_size, kernel_size, inp_channel, op_channel],
                                 initializer=tf.keras.initializers.he_normal())
        b_conv = tf.get_variable("b_" + name, initializer=tf.zeros(op_channel))
        z_conv = tf.nn.conv2d(x_padded, W_conv, strides=[1, strides, strides, 1], padding=padding) + b_conv
        a_conv = tf.nn.relu(z_conv)
        h_conv = tf.layers.batch_normalization(a_conv, training = self._is_training, renorm = True)
        if dropout:
            a_conv_dropout = tf.nn.dropout(a_conv, keep_prob=self._keep_prob)
            return a_conv_dropout
        if not_activated:
            return z_conv
        return h_conv

    def convolutional_layer_pretrained(self, x, filter, bias):
        W_conv = tf.constant(filter)
        b_conv = tf.constant(bias)
        return tf.nn.conv2d(x, W_conv, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv

    def depthwise_separable_conv_layer(self, x, name, inp_channel, op_channel, depth_kernel,
                                       depth_multiplier = 1, strides = 1, pad = 0, padding = 'SAME'):
        if pad != 0:
            x_padded = tf.pad(x, self.create_pad(4, pad))
        else:
            x_padded = x

        depth_filter = tf.get_variable(
            name = name + "_depth_kernel",
            shape = [depth_kernel, depth_kernel, inp_channel, depth_multiplier]
        )
        point_filter = tf.get_variable(
            name = name + "point_filter",
            shape = [1, 1, inp_channel * depth_multiplier, op_channel]
        )
        separable_conv = tf.nn.separable_conv2d(
            input = x,
            depthwise_filter = depth_filter,
            pointwise_filter = point_filter,
            strides = [1, strides, strides, 1],
            padding = padding,
            name = name + "_separable_conv"
        )
        return separable_conv


    def deconvolutional_layer(self, x, name, inp_shape, op_shape, kernel_size = 3, strides = 1, padding='VALID'):
        b_deconv = tf.get_variable("b" + name, initializer=tf.zeros(op_shape[3]))
        filter = tf.get_variable("filter" + name, shape=[kernel_size, kernel_size, op_shape[3], inp_shape[3]])
        z_deconv = tf.nn.conv2d_transpose(x, filter=filter, strides=[1, strides, strides, 1], padding=padding,
                                          output_shape=tf.stack(op_shape)) + b_deconv
        a_deconv = tf.nn.relu(z_deconv)
        h_conv = tf.layers.batch_normalization(a_deconv, training = self._is_training, renorm = True)
        return z_deconv


    def convolutional_module_with_max_pool(self, x, inp_channel, op_channel, name, strides = 1):
        # conv1 = self.convolutional_layer(x, inp_channel = inp_channel, op_channel = op_channel, name = name + "_conv1")
        conv1 = self.convolutional_layer(x, inp_channel=inp_channel, op_channel=op_channel, name=name + "_conv1", strides = strides)
        conv2 = self.convolutional_layer(conv1, inp_channel=op_channel, op_channel=op_channel, name=name + "_conv2", strides = strides)
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
        # conv3 = self.convolutional_layer(conv2, name + "conv3", inp_channel, op_channel, dropout = True)
        res_layer = tf.nn.relu(tf.add(conv2, x, name="res"))

        batch_norm = tf.contrib.layers.batch_norm(res_layer, is_training=self._is_training, renorm = True)

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
            op_channel = inp_channel
        )
        sep_conv_1_norm = tf.layers.batch_normalization(sep_conv_1, training = self._is_training)
        sep_conv_1_norm_activated = tf.nn.relu(sep_conv_1_norm)

        sep_conv_2 = self.depthwise_separable_conv_layer(
            sep_conv_1_norm_activated,
            name=name + "sep_conv_2",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel = inp_channel
        )
        sep_conv_2_norm = tf.layers.batch_normalization(sep_conv_2, training = self._is_training)
        sep_conv_2_norm_activated = tf.nn.relu(sep_conv_2_norm)

        sep_conv_3 = self.depthwise_separable_conv_layer(
            sep_conv_2_norm_activated,
            name=name + "sep_conv_3",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel = inp_channel
        )
        sep_conv_3_norm = tf.layers.batch_normalization(sep_conv_3, training = self._is_training, renorm = True)

        res = tf.nn.relu(x + sep_conv_3_norm)
        return tf.layers.batch_normalization(res, training = self._is_training)

    def squeeze(self, x):
        return self.global_average_pooling(x)

    def excite(self, x, name, n_channels, reduction_ratio = 16):
        x_shape = tf.shape(x)
        W_1 = tf.get_variable(shape = [n_channels, n_channels // reduction_ratio], name = name + "_W1")
        z_1 = tf.nn.relu(tf.matmul(x, W_1))
        W_2 = tf.get_variable(shape = [n_channels // reduction_ratio, n_channels], name = name + "_W2")
        return tf.nn.sigmoid(tf.matmul(z_1, W_2))


    def se_block(self, x, name, n_channels):
        x_shape = tf.shape(x)
        x_squeezed = self.squeeze(x)
        x_excited = self.excite(x_squeezed, name = name + "_excited", n_channels = n_channels)
        x_excited_broadcasted = tf.reshape(x_excited, shape = [x_shape[0], 1, 1, x_shape[-1]])
        return tf.multiply(x, x_excited_broadcasted)

    def residual_module_with_se(self, x, name, inp_channel):
        conv1 = self.convolutional_layer(x, name + "_conv1", inp_channel, inp_channel)
        conv2 = self.convolutional_layer(conv1, name + "_conv2", inp_channel, inp_channel, not_activated=True)
        conv2_se = self.se_block(conv2, name = name + "_se", n_channels = inp_channel)
        res_layer = tf.nn.relu(tf.add(conv2_se, x, name = name + "res"))
        batch_norm = tf.layers.batch_normalization(res_layer, training = self._is_training, renorm = True)
        return batch_norm

    def xception_module_with_se(self, x, name, inp_channel):
        # x_activated = tf.nn.relu(x)
        sep_conv_1 = self.depthwise_separable_conv_layer(
            x,
            name=name + "sep_conv_1",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel = inp_channel
        )
        sep_conv_1_norm = tf.layers.batch_normalization(sep_conv_1, training = self._is_training)
        sep_conv_1_norm_activated = tf.nn.relu(sep_conv_1_norm)

        sep_conv_2 = self.depthwise_separable_conv_layer(
            sep_conv_1_norm_activated,
            name=name + "sep_conv_2",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel = inp_channel
        )
        sep_conv_2_norm = tf.layers.batch_normalization(sep_conv_2, training = self._is_training)
        sep_conv_2_norm_activated = tf.nn.relu(sep_conv_2_norm)

        sep_conv_3 = self.depthwise_separable_conv_layer(
            sep_conv_2_norm_activated,
            name=name + "sep_conv_3",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel = inp_channel
        )
        sep_conv_3_norm = tf.layers.batch_normalization(sep_conv_3, training = self._is_training)
        sep_conv_3_norm_se = self.se_block(sep_conv_3_norm, name = name + "_se", n_channels = inp_channel)

        res = tf.nn.relu(x + sep_conv_3_norm_se)
        batch_norm = tf.layers.batch_normalization(res, training = self._is_training, renorm = True)
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

    def instance_norm(self, x, epsilon = 1e-8):
        batch_size = tf.shape(x)[0]
        n_channel = tf.shape(x)[3]

        mean = tf.reduce_mean(tf.reduce_mean(x, axis = 1), axis = 1)
        mean_expanded = tf.reshape(mean, shape = [batch_size, 1, 1, n_channel])
        var = tf.reduce_mean(tf.reduce_mean(tf.square(x - mean_expanded), axis = 1), axis = 1)
        var_expanded = tf.reshape(var, shape = [batch_size, 1, 1, n_channel])

        normalized = (x - mean_expanded) / tf.sqrt(var_expanded + epsilon)
        scale = tf.Variable(tf.ones([n_channel]))
        shift = tf.Variable(tf.zeros([n_channel]))

        return scale * normalized + shift

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def global_average_pooling(self, x):
        return tf.reduce_mean(x, axis=[1, 2])

    # Predict:
    def predict(self, X, batch_size = None):
        if batch_size is None:
            batch_size = X.shape[0]
        train_indicies = np.arange(X.shape[0])
        predictions = np.zeros(shape = [X.shape[0], self._img_h, self._img_w, self._img_c])

        for i in range(int(math.ceil(X.shape[0] / batch_size))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % X.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            # get batch size
            actual_batch_size = X[idx, :].shape[0]

            op = self._sess.run(self._X_transformed, feed_dict={
                self._X: X[idx, :],
                self._batch_size: actual_batch_size,
                self._is_training: False,
                self._keep_prob_tensor: 1.0})
            predictions[idx, :, :, :] = op

        return predictions


    # Train:
    def fit(self, X, y, num_epoch = 1, batch_size = 16, weight_save_path=None, weight_load_path=None,
            plot_losses=False, print_every = 1):
        self._optimizer = tf.train.AdamOptimizer(1e-4)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self._train_step = self._optimizer.minimize(self._mean_loss)
        self._sess = tf.Session()
        if weight_load_path is not None:
            loader = tf.train.Saver()
            loader.restore(sess=self._sess, save_path=weight_load_path)
            print("Weight loaded successfully")
        else:
            self._sess.run(tf.global_variables_initializer())
        if num_epoch > 0:
            print('Training Denoising Net for ' + str(num_epoch) + ' epochs')
            self.run_model(self._sess, X, y, num_epoch, batch_size, print_every,
                           self._train_step, weight_save_path=weight_save_path, plot_losses=plot_losses)

        # Adapt from Stanford's CS231n Assignment3
    def run_model(self, session, Xd, yd,
                  epochs=1, batch_size=1, print_every=1,
                  training=None, plot_losses=False, weight_save_path=None, patience=None):
        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self._mean_loss]
        if training_now:
            variables[-1] = training
            self._keep_prob_passed = self._keep_prob
        else:
            self._keep_prob_passed = 1.0

        # counter
        iter_cnt = 0
        val_losses = []
        early_stopping_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
                # generate indicies for the batch
                start_idx = (i * batch_size) % Xd.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                # create a feed dictionary for this batch
                # get batch size
                actual_batch_size = yd[idx].shape[0]

                if i < int(math.ceil(Xd.shape[0] / batch_size)) - 1:
                    feed_dict = {self._X: Xd[idx, :],
                                 self._is_training: training_now,
                                 self._batch_size: actual_batch_size,
                                 self._keep_prob_tensor: self._keep_prob_passed}
                    # have tensorflow compute loss and correct predictions
                    # and (if given) perform a training step
                    loss, corr, _ = session.run(variables, feed_dict=feed_dict)

                    # aggregate performance stats
                    losses.append(loss * actual_batch_size)
                    correct += np.sum(corr)

                    # print every now and then
                    if training_now and (iter_cnt % print_every) == 0:
                        print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                              .format(iter_cnt, loss, np.sum(corr) / actual_batch_size / self._img_h / self._img_w))


                else:
                    feed_dict = {self._X: Xd[idx, :],
                                 self._is_training: False,
                                 self._batch_size: actual_batch_size,
                                 self._keep_prob_tensor: 1.0}
                    val_loss = session.run(self._mean_loss, feed_dict=feed_dict)
                    print("Validation loss: " + str(val_loss))
                    val_losses.append(val_loss)
                    # if training_now and weight_save_path is not None:
                    if training_now and val_loss <= min(val_losses) and weight_save_path is not None:
                        save_path = self._saver.save(session, save_path=weight_save_path)
                        print("Model's weights saved at %s" % save_path)
                    if patience is not None:
                        if val_loss > min(val_losses):
                            early_stopping_cnt += 1
                        else:
                            early_stopping_cnt = 0
                        if early_stopping_cnt > patience:
                            print("Patience exceeded. Finish training")
                            return
                iter_cnt += 1
            total_correct = correct / (Xd.shape[0] - actual_batch_size)
            total_loss = np.sum(losses) / (Xd.shape[0] - actual_batch_size)
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
                  .format(total_loss, total_correct / self._img_h / self._img_w, e + 1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e + 1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        return total_loss, total_correct


    def create_pad(self, n, pad):
        pad_matrix = [[0, 0]]
        for i in range(n - 2):
            pad_matrix.append([pad, pad])
        pad_matrix.append([0, 0])
        return tf.constant(pad_matrix)

    def save_weights(self, weight_save_path):
        self._saver.save(sess = self._sess, save_path = weight_save_path)
        print("Weight saved successfully")

    # def evaluate(self, X, y):
    #     self.run_model(self._sess, self._op, self._mean_loss, X, y, 1, 16)

    def losses(self, img, target, layers, loss_type, path):
        img_net = self.vgg_net(img, path)
        target_net = self.vgg_net(target, path)

        loss = 0
        for layer in layers:
            loss += loss_type(img_net[layer], target_net[layer])

        return loss

    def feat_loss(self, img, content_img):
        h = tf.shape(img)[1]
        w = tf.shape(img)[2]
        n_channel = tf.shape(img)[3]

        square_dif = tf.square(img - content_img)

        return tf.reduce_sum(square_dif) / tf.cast(h, tf.float32) / tf.cast(w, tf.float32) / tf.cast(n_channel, tf.float32)

    def style_loss(self, img, style_img):
        img_graham = self.graham_mats(img)
        style_img_graham = self.graham_mats(style_img)

        diff = img_graham - style_img_graham
        square_dif = diff * diff

        sum_square_diff = tf.reduce_sum(tf.reduce_sum(square_dif, axis = -1), axis = -1)
        return tf.reduce_mean(sum_square_diff)

    def style_loss_v2(self, img_transformed, style_img):
        img_graham = self.graham_tensor(img_transformed)
        style_img_graham = self.graham_tensor(style_img)

        diff = img_graham - style_img_graham
        square_dif = diff * diff

        sum_square_diff = tf.reduce_sum(tf.reduce_sum(square_dif, axis = -1), axis = -1)
        return tf.reduce_mean(tf.diag_part(sum_square_diff))

    def graham_tensor(self, img):
        h = tf.shape(img)[1]
        w = tf.shape(img)[2]
        n_channel = tf.shape(img)[3]

        img_reshaped = tf.reshape(img, shape = [-1, h * w, n_channel])
        img_transposed = tf.transpose(img_reshaped, perm = [0, 2, 1])
        graham_tensor = tf.tensordot(img_reshaped, img_transposed, axes = [[1], [2]])
        graham_tensor_transposed = tf.transpose(graham_tensor, perm = [0, 2, 1, 3])


        return graham_tensor_transposed / tf.cast(h, tf.float32) / tf.cast(w, tf.float32) / tf.cast(n_channel, tf.float32)

    def graham_mats(self, img):
        n_batch = tf.shape(img)[0]
        h = tf.shape(img)[1]
        w = tf.shape(img)[2]
        n_channel = tf.shape(img)[3]

        img_reshaped = tf.reshape(img, shape = [-1, h * w, n_channel])
        img_transposed = tf.transpose(img_reshaped, perm = [0, 2, 1])
        graham_tensor = tf.tensordot(img_reshaped, img_transposed, axes = [[1], [2]])
        graham_tensor_transposed = tf.transpose(graham_tensor, perm = [0, 2, 1, 3])

        mask = tf.reshape(tf.eye(num_rows = n_batch), shape = [n_batch, n_batch, 1, 1])

        graham_mats = tf.reduce_sum(graham_tensor_transposed * mask, axis = 1)

        return graham_mats / tf.cast(h, tf.float32) / tf.cast(w, tf.float32) / tf.cast(n_channel, tf.float32)

    # https://github.com/lengstrom/fast-style-transfer/blob/master/src/vgg.py
    def vgg_net(self, img, vgg_path):
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )

        data = scipy.io.loadmat(vgg_path)
        weights = data['layers'][0]
        net = {}

        current = img
        for ind, layer in enumerate(layers):
            kind = layer[:4]

            if kind == 'conv':
                filter, bias = weights[ind][0][0][0][0]

                # Reshaping the weights:
                ## matconvnet: weights are [width, height, in_channels, out_channels]
                ## tensorflow: weights are [height, width, in_channels, out_channels]
                filter = np.transpose(filter, (1, 0, 2, 3))
                bias = bias.reshape(-1)

                current = self.convolutional_layer_pretrained(current, filter, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current)
            elif kind == 'pool':
                current = self.max_pool_2x2(current)

            net[layer] = current

        return net




