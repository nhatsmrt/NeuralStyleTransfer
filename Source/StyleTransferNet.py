'''
Note that the img is processed in BGR space (same for vgg net)
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io
import cv2
class StyleTransferNet():
    def __init__(
            self, style_img, keep_prob = 0.9, style_weight = 1, content_weight = 4E5, tv_weight = 1,
            is_training = True, pretrained_path = None, MEAN_PIXEL = np.array([103.939, 116.779, 123.68])):

        if is_training:
            self._style_img = np.array([style_img])
            self._keep_prob = keep_prob
            self._pretrained_path = pretrained_path
            self._style_weight = style_weight
            self._content_weight = content_weight
            self._tv_weight = tv_weight

        self._MEAN_PIXEL = MEAN_PIXEL
        self._use_gpu = True

        with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
            self.build(is_training)


    def build(self, is_training):
        self._sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        CONTENT_LAYERS = ['conv5_2']

        # Build image transformation net:
        self.image_transformation_net()
        save_list = tf.trainable_variables()
        self._saver = tf.train.Saver(save_list)


        if is_training:
            # Precompute style:
            self._style_features = {}
            with tf.device('/device:CPU:0'):
                self._style_img_pre = self.preprocess(self._style_img, is_tensor = False)
                self._style_vgg = self.vgg_net(self._style_img_pre, vgg_path = self._pretrained_path)
                for layer in STYLE_LAYERS:
                    style_feature = self._sess.run(self._style_vgg[layer])
                    # Compute the Gram matrix:
                    style_feature = np.reshape(style_feature, (-1, style_feature.shape[3]))
                    gram = np.matmul(style_feature.T, style_feature) / (style_feature.size * 2)
                    self._style_features[layer] = gram
                print("Finish precomputing style features' gram matrices")

            # with tf.device('/device:GPU:0'):
            self._X_transformed_pre = self.preprocess(self.image_transformation_net())
            self._X_transformed_vgg = self.vgg_net(self._X_transformed_pre, vgg_path = self._pretrained_path)
            self._X_vgg = self.vgg_net(self._X_pre, vgg_path = self._pretrained_path)

            self._style_loss = self.losses(
                self._X_transformed_vgg,
                self._style_features,
                layers = STYLE_LAYERS,
                loss_type = self.style_loss,
            )
            self._feat_loss = self.losses(
                self._X_transformed_vgg,
                self._X_vgg,
                layers = CONTENT_LAYERS,
                loss_type = self.feat_loss,
            )
            self._tv_loss = self.total_variation_regularizer(self._pred)
            self._mean_loss = self._style_weight * self._style_loss / len(STYLE_LAYERS) + \
                              self._content_weight * self._feat_loss / len(CONTENT_LAYERS) + \
                              self._tv_weight * self._tv_loss

            self._optimizer = tf.train.AdamOptimizer(1e-3)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                self._train_step = self._optimizer.minimize(self._mean_loss)

        self._init_op = tf.global_variables_initializer()
        self._sess = tf.Session()
        # self._saver = tf.train.Saver()



    def image_transformation_net(self):
        self._X = tf.placeholder(shape = [None, None, None, None], dtype = tf.float32)
        self._batch_size = tf.shape(self._X)[0]
        self._img_h = tf.shape(self._X)[1]
        self._img_w = tf.shape(self._X)[2]
        self._n_channel = tf.shape(self._X)[3]
        self._is_training = tf.placeholder(tf.bool)
        self._keep_prob_tensor = tf.placeholder(tf.float32)


        self._X_pre = self.preprocess(self._X)
        # self._X_norm = tf.layers.batch_normalization(self._X, training=self._is_training)
        self._X_norm = self.instance_norm(self._X_pre / 255)

        self._conv_layer_1 = self.convolutional_module(
            x = self._X_norm,
            inp_channel = 3,
            op_channel = 4,
            name = "module_1",
        )

        self._res_1 = self.residual_module(self._conv_layer_1, name = "res_1", inp_channel = 4)
        self._res_2 = self.residual_module(self._res_1, name = "res_2", inp_channel = 4)
        self._res_3 = self.residual_module(self._res_2, name = "res_3", inp_channel = 4)
        self._res_4 = self.residual_module(self._res_3, name = "res_4", inp_channel = 4)

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
        self._resized = self.resize_convolution_layer(
            self._res_4,
            new_h = self._img_h,
            new_w = self._img_w,
            inp_channel = 4,
            op_channel = 3,
            name = "resized"
        )
        self._pred = self._resized * 255
        return self._pred





    # Define layers and modules:
    def convolutional_layer(self, x, name, inp_channel, op_channel, kernel_size=3, strides = 1, padding='VALID',
                            pad = 1, dropout=False, not_activated=False, not_normed = False):
        if pad != 0:
            x_padded = tf.pad(x, self.create_pad(4, pad))
        else:
            x_padded = x
        W_conv = tf.get_variable("W_" + name, shape=[kernel_size, kernel_size, inp_channel, op_channel],
                                 initializer=tf.keras.initializers.he_normal())
        b_conv = tf.get_variable("b_" + name, initializer=tf.zeros(op_channel))
        z_conv = tf.nn.conv2d(x_padded, W_conv, strides=[1, strides, strides, 1], padding=padding) + b_conv
        a_conv = tf.nn.relu(z_conv)
        # h_conv = tf.layers.batch_normalization(a_conv, training = self._is_training, renorm = True)
        if dropout:
            a_conv_dropout = tf.nn.dropout(a_conv, keep_prob=self._keep_prob)
            return a_conv_dropout
        if not_activated:
            return z_conv
        if not_normed:
            return a_conv

        h_conv = self.instance_norm(a_conv, n_channel = op_channel)
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


    def deconvolutional_layer(self, x, name, inp_shape, op_shape, kernel_size = 3, strides = 1, padding='VALID', activated = False):
        b_deconv = tf.get_variable("b" + name, initializer=tf.zeros(op_shape[3]))
        filter = tf.get_variable("filter" + name, shape=[kernel_size, kernel_size, op_shape[3], inp_shape[3]])
        z_deconv = tf.nn.conv2d_transpose(x, filter=filter, strides=[1, strides, strides, 1], padding=padding,
                                          output_shape=tf.stack(op_shape)) + b_deconv
        if activated:
            a_deconv = tf.nn.relu(z_deconv)
            h_deconv = self.instance_norm(a_deconv, n_channel = op_shape[3])
            return h_deconv
        
        return z_deconv

    def resize_convolution_layer(self, x, name, new_h, new_w, inp_channel, op_channel):
        x_resized = tf.image.resize_images(x, size = (new_h, new_w))
        x_resized_conv = tf.nn.sigmoid(self.convolutional_layer(
            x_resized,
            inp_channel = inp_channel,
            op_channel = op_channel,
            name = name + "_conv",
            not_activated = True
        ))
        return x_resized_conv

    def convolutional_module(self, x, inp_channel, op_channel, name):
        conv1 = self.convolutional_layer(x, inp_channel = inp_channel, op_channel=op_channel, name = name + "_conv1")
        conv2 = self.convolutional_layer(conv1, inp_channel = op_channel, op_channel=op_channel, name = name + "_conv2", strides = 2)
        return conv2


    def convolutional_module_with_max_pool(self, x, inp_channel, op_channel, name, strides = 1):
        conv1 = self.convolutional_layer(x, inp_channel = inp_channel, op_channel=op_channel, name=name + "_conv1", strides = strides)
        conv2 = self.convolutional_layer(conv1, inp_channel = op_channel, op_channel=op_channel, name=name + "_conv2", strides = strides)
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

        # batch_norm = tf.contrib.layers.batch_norm(res_layer, is_training=self._is_training, renorm = True)
        batch_norm = self.instance_norm(res_layer, n_channel = inp_channel)

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
            # a_norm = tf.layers.batch_normalization(a, training=self._is_training)
            a_norm = self.instance_norm(a, n_channel = op_channel)
            return a_norm

    def instance_norm(self, x, n_channel = 3, epsilon = 1e-8):
        batch_size = tf.shape(x)[0]
        # n_channel = tf.shape(x)[3]

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
        if len(X.shape) == 4:
            train_indicies = np.arange(X.shape[0])
            predictions = np.zeros(shape = X.shape)
            for i in range(int(math.ceil(X.shape[0] / batch_size))):
                # generate indicies for the batch
                start_idx = (i * batch_size) % X.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                # create a feed dictionary for this batch
                # get batch size

                op = self._sess.run(self._pred, feed_dict={
                    self._X: X[idx, :],
                    self._is_training: False,
                    self._keep_prob_tensor: 1.0})
                predictions[idx, :, :, :] = op
            return np.round(predictions).astype(np.uint8)
        else:
            predictions = []
            for img in X:
                img = np.array([img])
                prediction = self._sess.run(self._pred, feed_dict={
                    self._X: img,
                    self._is_training: False,
                    self._keep_prob_tensor: 1.0})
                predictions.append(np.round(prediction[0]).astype(np.uint8))
            return np.array(predictions)

    # Train:
    def fit(self, X, X_val = None, num_epoch = 1, batch_size = 16, patience = None, weight_save_path=None, weight_load_path=None,
            plot_losses=False, draw_img = False, print_every = 1):
        if weight_load_path is not None:
            self._sess.run(self._init_op)
            self._saver.restore(sess=self._sess, save_path=weight_load_path)
            print("Weight loaded successfully")
        else:
            self._sess.run(tf.global_variables_initializer())
        if num_epoch > 0:
            print('Training Style Transfer Net for ' + str(num_epoch) + ' epochs')
            self.run_model(self._sess, X, X_val, num_epoch, batch_size, print_every,
                           self._train_step, patience = patience, weight_save_path=weight_save_path,
                           plot_losses=plot_losses, draw_img = draw_img)

    # Adapt from Stanford's CS231n Assignment3
    def run_model(self, session, Xd, X_val = None,
                  epochs=1, batch_size=1, print_every=1,
                  training=None, plot_losses=False, draw_img = False, weight_save_path=None, patience=None):
        # shuffle indicies

        training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self._mean_loss]
        if training_now:
            variables.append(training)
            self._keep_prob_passed = self._keep_prob
        else:
            self._keep_prob_passed = 1.0

        # counter
        iter_cnt = 0
        val_losses = []
        early_stopping_cnt = 0
        for e in range(epochs):
            print("Epoch " + str(e + 1))
            train_indicies = np.arange(Xd.shape[0])
            np.random.shuffle(train_indicies)
            # keep track of losses and accuracy
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
                # generate indicies for the batch
                start_idx = (i * batch_size) % Xd.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                # create a feed dictionary for this batch
                # get batch size
                actual_batch_size = Xd[idx, :].shape[0]

                feed_dict = {self._X: Xd[idx, :],
                             self._is_training: training_now,
                             self._keep_prob_tensor: self._keep_prob_passed}
                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, _ = session.run(variables, feed_dict=feed_dict)
                # print(session.run(self._style_loss, feed_dict = feed_dict))

                # aggregate performance stats
                losses.append(loss * actual_batch_size)

                # print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g}" \
                          .format(iter_cnt, loss))
                iter_cnt += 1

            #  Validate and/or save weights
            if X_val is not None:
                feed_dict = {self._X: X_val,
                             self._is_training: False,
                             self._keep_prob_tensor: 1.0}
                val_loss = session.run(self._mean_loss, feed_dict=feed_dict)
                feat_loss = session.run(self._feat_loss, feed_dict=feed_dict)
                st_loss = session.run(self._style_loss, feed_dict=feed_dict)
                tv_loss = session.run(self._tv_loss, feed_dict=feed_dict)
                print("Validation loss: " + str(val_loss))
                print("Content loss: " + str(feat_loss))
                print("Style loss: " + str(st_loss))
                print("TV loss: " + str(tv_loss))

                val_losses.append(val_loss)

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

                if draw_img:
                    fig = plt.figure()
                    a = fig.add_subplot(2, 2, 1)
                    plt.imshow(cv2.cvtColor(X_val[0], cv2.COLOR_BGR2RGB))
                    a = fig.add_subplot(2, 2, 2)
                    prediction = self.predict(X_val[:1])[0]
                    plt.imshow(cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB))
                    a = fig.add_subplot(2, 2, 3)
                    plt.imshow(cv2.cvtColor(X_val[1], cv2.COLOR_BGR2RGB))
                    a = fig.add_subplot(2, 2, 4)
                    prediction_2 = self.predict(X_val[1:2])[0]
                    plt.imshow(cv2.cvtColor(prediction_2, cv2.COLOR_BGR2RGB))
                    plt.show()
            else:
                save_path = self._saver.save(session, save_path=weight_save_path)
                print("Model's weights saved at %s" % save_path)

            total_loss = np.sum(losses) / (Xd.shape[0])
            print("Epoch {1}, Overall loss = {0:.3g}" \
                  .format(total_loss, e + 1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e + 1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        return total_loss


    def create_pad(self, n, pad):
        pad_matrix = [[0, 0]]
        for i in range(n - 2):
            pad_matrix.append([pad, pad])
        pad_matrix.append([0, 0])
        return tf.constant(pad_matrix)

    def save_weights(self, weight_save_path):
        self._saver.save(sess = self._sess, save_path = weight_save_path)
        print("Weights saved successfully")

    def load_weights(self, weight_load_path):
        self._sess.run(self._init_op)
        self._saver.restore(self._sess, weight_load_path)
        print("Weights loaded successfully")


    # def evaluate(self, X, y):
    #     self.run_model(self._sess, self._op, self._mean_loss, X, y, 1, 16)

    def losses(self, img_net, target_net, layers, loss_type):
        loss = 0
        for layer in layers:
            loss += loss_type(img_net[layer], target_net[layer])
        return loss

    def total_variation_regularizer(self, img):
        batch_size = tf.cast(tf.shape(img)[0], tf.float32)
        h = tf.shape(img)[1]
        w = tf.shape(img)[2]
        n_channel = tf.shape(img)[3]

        h_size = tf.cast((h - 1) * w * n_channel, tf.float32)
        w_size = tf.cast(h * (w - 1) * n_channel, tf.float32)

        img_dh = img[:, 1:, :, :] - img[:, :h - 1, :, :]
        img_dw = img[:, :, 1:, :] - img[:, :, :w - 1, :]
        img_tv_h = tf.nn.l2_loss(img_dh)
        img_tv_w = tf.nn.l2_loss(img_dw)

        return (img_tv_h / h_size + img_tv_w / w_size) / batch_size



    def feat_loss(self, img, content_img):
        square_dif = tf.square(img - content_img)
        return tf.reduce_mean(square_dif)

    def style_loss(self, img, style_img_gram):
        img_gram = self.gram_mats(img) # batch_size x n_channel x n_channel

        diff = img_gram - np.expand_dims(style_img_gram, 0)
        square_dif = tf.square(diff) # batch_size x n_channel x n_channel

        sum_square_diff = tf.reduce_sum(tf.reduce_sum(square_dif, axis = -1), axis = -1) # batch_size
        return tf.reduce_mean(sum_square_diff)


    def gram_mats(self, img):
        # return shape: batch_size x n_channel x n_channel
        h = tf.shape(img)[1]
        w = tf.shape(img)[2]
        n_channel = tf.shape(img)[3]

        img_reshaped = tf.reshape(img, shape = [-1, h * w, n_channel])
        img_transposed = tf.transpose(img_reshaped, perm = [0, 2, 1])
        return tf.matmul(img_transposed, img_reshaped) / (2 * tf.cast(h, tf.float32) * tf.cast(w, tf.float32) * tf.cast(n_channel, tf.float32))

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
    
    def preprocess(self, img, is_tensor = True):
        print("Preprocess image.")
        if is_tensor:
            return (img - self._MEAN_PIXEL)
        else:
            return (img - self._MEAN_PIXEL).astype(np.float32)
    
    




