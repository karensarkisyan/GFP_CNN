
# coding: utf-8

# In[1]:


from functions import *


# In[13]:


class Data():
    
    def __init__(self,file_path,batch_size,zero_sample_fraction, zeroing = False):
        
        self.initial_df, self.input_df, self.mutant_list, self.labels, self.sample_weights = load_data(file_path)
        
        self.batch_size = batch_size
        self.zero_sample_fraction = zero_sample_fraction
        
        if zeroing:
            self.labels[self.labels<2.7]=0
            
        self.x_train, self.x_val, self.idx_y_train, self.idx_y_val = train_test_split(self.input_df, 
                                                                                      [num for num in range(len(self.labels))], 
                                                                                      train_size=0.9)
        self.y_train = self.labels[self.idx_y_train]
        self.y_val = self.labels[self.idx_y_val]
        self.sample_weights_train = self.sample_weights[self.idx_y_train]
        self.sample_weights_val = self.sample_weights[self.idx_y_val]

        self.all_inds = np.array([p for p in range(len(self.x_train))])
        self.zero_inds = self.all_inds[self.y_train.flatten()==0]
        self.nonzero_inds = self.all_inds[self.y_train.flatten()!=0]
        
    def plot_labels_distribution(self):
        plt.figure(figsize=[10,8])
        plt.hist(self.labels,bins=100,color='k',alpha=0.5)


# In[10]:


class ResNet(object):
    
    @add_arg_scope
    def ResNet_architecture(net, x, kernel_size, pool_size, weight_decay, keep_prob):

        temp = bn(x, name='bn_first')
        temp = conv(temp, 64, kernel_size=kernel_size, padding='SAME', 
                    kernel_initializer=xavier(), kernel_regularizer=l2_reg(weight_decay),
                   name='conv_first')
        temp = max_pool(inputs=temp, pool_size=pool_size, strides=2,padding='SAME')

        for scale in range(net.num_scales):
            with tf.variable_scope("scale%i" % scale):
                for rep in range(net.block_repeats):
                    with tf.variable_scope("rep%i" % rep):
                        temp = residual_block(temp, kernel_size, weight_decay)

                if scale < net.num_scales - 1:
                    temp = conv(temp, 2 * x.get_shape().as_list()[-1],
                             kernel_size=kernel_size, strides=2,
                             padding='SAME', name='conv_downsample', kernel_regularizer=l2_reg(weight_decay))

        temp = avg_pool(temp, pool_size=pool_size, strides=2)

        temp = bn(temp, name='bn_last')
        temp = tf.nn.relu(temp)

        temp = tf.contrib.layers.flatten(temp)

        temp = tf.nn.dropout(temp, keep_prob=keep_prob)
        temp = dense(temp,1000, activation=tf.nn.relu,name='dense_1', kernel_regularizer=l2_reg(weight_decay))
        temp = tf.nn.dropout(temp, keep_prob=keep_prob)
        temp = dense(temp,1000, activation=tf.nn.relu,name='dense_2', kernel_regularizer=l2_reg(weight_decay))
        temp = tf.nn.dropout(temp, keep_prob=keep_prob)
        temp = dense(temp,100, activation=tf.nn.relu,name='dense_3', kernel_regularizer=l2_reg(weight_decay))
        temp = tf.nn.dropout(temp, keep_prob=keep_prob)
        temp = dense(temp,1, activation=tf.nn.relu,name='dense_4', kernel_regularizer=l2_reg(weight_decay))

        return temp
    
    def __init__(self, input_data, num_scales, block_repeats, NN_name, mode,
                kernel_size, pool_size, weight_decay, keep_prob, n_epoch):
        
        self.num_scales = num_scales
        self.block_repeats = block_repeats
        self.name = NN_name
        self.mode = mode
        self.n_epoch = n_epoch
        
        self.x_train_ph = tf.placeholder(dtype=tf.float32,shape=[None,238,21],name='train_X_ph')
        self.y_train_ph = tf.placeholder(dtype=tf.float32,shape=[None,1],name='train_Y_ph')

        self.x_val_ph = tf.placeholder(dtype=tf.float32,shape=[None,238,21],name='val_X_ph')
        self.y_val_ph = tf.placeholder(dtype=tf.float32,shape=[None,1],name='val_Y_ph')

        self.sample_weights_ph = tf.placeholder(dtype=tf.float32,shape=[None,1],name='sample_weights_ph')
        
        with tf.variable_scope(self.name):

            with tf.device('/%s:0' % mode):
                with arg_scope([conv]):
                    with arg_scope([residual_block], keep_prob=keep_prob):
                        with arg_scope([bn], training=True):
                            with arg_scope([dense]):
                                self.preds_train = ResNet_architecture(self, self.x_train_ph, 
                                                                   kernel_size, pool_size, 
                                                                   weight_decay, keep_prob)

                with arg_scope([conv], reuse=True):
                    with arg_scope([dense],reuse=True):
                        with arg_scope([residual_block], keep_prob=1):
                            with arg_scope([bn], training=False, reuse=True):
                                self.preds_val = ResNet_architecture(self, self.x_val_ph, 
                                                                   kernel_size, pool_size, 
                                                                   weight_decay, keep_prob=1)
        
        self.loss = tf.losses.mean_squared_error(self.y_train_ph,self.preds_train)
        self.loss_weighted = tf.reduce_mean(tf.squared_difference(self.y_train_ph, self.preds_train)*self.sample_weights_ph)
        self.loss_reg = [self.loss_weighted] + tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        self.val_loss = tf.losses.mean_squared_error(self.y_val_ph,self.preds_val)
        
        self.learning_rate = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
        self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        self.train_step = self.opt.minimize(self.loss_reg)
        self.train_step = tf.group(*([self.train_step] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)))
        
        self.init = tf.global_variables_initializer()

