import tensorflow as tf
import numpy as np

class NAE():


    def __init__(self,n_input, dimensions, learning_rate=0.005, activation=tf.nn.relu, bias = False, l2scale = 0.01, meaninit=0.0,stddev=0.05,identity_initializer=False):

        tf.reset_default_graph()

        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, n_input])

        # let's first copy our X placeholder to the name current_input
        current_input = X

        # We're going to keep every matrix we create so let's create a list to hold them all
        Ws = []
        bs = []
        activs =[]

        # We'll create a for loop to create each layer:
        for layer_i, n_output in enumerate(dimensions):

            # just like in the last session,
            # we'll use a variable scope to help encapsulate our variables
            # This will simply prefix all the variables made in this scope
            # with the name we give it.
            with tf.variable_scope("encoder/layer/{}".format(layer_i)):

                # Create a weight matrix which will increasingly reduce
                # down the amount of information in the input by performing
                # a matrix multiplication
                if identity_initializer:

                    
                    W = tf.get_variable(
                        name='W',
                        shape=[n_input, n_output],
                        initializer=tf.constant_initializer(np.identity(n_input)*meaninit),
                        regularizer = tf.contrib.layers.l2_regularizer(l2scale))
                else:

                    W = tf.get_variable(
                        name='W',
                        shape=[n_input, n_output],
                        initializer=tf.random_normal_initializer(mean=meaninit, stddev=stddev),
                        regularizer = tf.contrib.layers.l2_regularizer(l2scale))

                # Now we'll multiply our input by our newly created W matrix
                # and add the bias
                h = tf.matmul(current_input, W)

                if bias:
                    
                    
                    b = tf.get_variable(name='b',shape=[n_output],initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                    regularizer = tf.contrib.layers.l2_regularizer(l2scale))
                    current_input = activation(tf.add(h,b))
                    bs.append(b)

                else:
                    
                    current_input = activation(h)
                

                
                # Finally we'll store the weight matrix so we can build the decoder.
                Ws.append(W)

                # We'll also replace n_input with the current n_output, so that on the
                # next iteration, our new number inputs will be correct.
                # print([n_input, n_output])

                n_input = n_output
                activs.append(current_input)
                
        # We'll first reverse the order of our weight matrices
        Ws = Ws[::-1]
        
        if bias:
            bs = bs[::-1]

        encoder_op = current_input

        # then reverse the order of our dimensions
        # appending the last layers number of inputs.
        dimensions = dimensions[::-1][1:] + [n_input]

        # print(dimensions)

        for layer_i, n_output in enumerate(dimensions):
            # we'll use a variable scope again to help encapsulate our variables
            # This will simply prefix all the variables made in this scope
            # with the name we give it.
            with tf.variable_scope("decoder/layer/{}".format(layer_i)):

                # Now we'll grab the weight matrix we created before and transpose it
                # So a 3072 x 784 matrix would become 784 x 3072
                # or a 256 x 64 matrix, would become 64 x 256
                W = tf.transpose(Ws[layer_i])

                if bias:
                    b = bs[layer_i]
                    # Now we'll multiply our input by our transposed W matrix
                    h = tf.matmul(tf.add(current_input,b), W)

                else:
                    h = tf.matmul(current_input, W)

                # And then use a relu activation function on its output
                current_input = activation(h)

                # We'll also replace n_input with the current n_output, so that on the
                # next iteration, our new number inputs will be correct.
                # print([n_input, n_output])

                n_input = n_output
                activs.append(current_input)


        Y = current_input

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)        
        reg = tf.reduce_mean(reg_losses)
        msd = tf.reduce_mean(tf.squared_difference(X, Y)) 
        cost = msd + reg

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        self.variables_dict = {'X':X,'Y':Y,'z':encoder_op,'cost':cost,'W':Ws,'msd':msd}
 
    def init_session(self):
        # Launch the graph
        config = tf.ConfigProto( device_count = {'GPU': 0} )
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self,data,batch_size, n_epochs,display_step=10 ):

        # Fit all training data
        self.costs_list = []
        self.msd_list = []
        total_batch = int(data.length/batch_size)
        for epoch_i in range(n_epochs):
            for batch_i in range(total_batch):
                batch_xs = data.next_batch(batch_size)

                self.sess.run(self.optimizer, feed_dict={self.variables_dict['X']: batch_xs })
                cost_value,msd_value = self.sess.run([self.variables_dict['cost'],self.variables_dict['msd']],  feed_dict={self.variables_dict['X']: batch_xs})
                
                self.costs_list.append(cost_value)
                self.msd_list.append(msd_value)
                    
            # Display logs per epoch step
            if epoch_i % display_step == 0:
                print("Epoch:", '%04d' % (epoch_i),"cost=", "{:.9f}".format(cost_value))

        print("Optimization Finished!")

        return self.costs_list, self.msd_list

    def save(self,filename):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, filename+".ckpt")
        print("Model saved in file: %s" % save_path)

    def load(self,filename):
        saver = tf.train.Saver()        
        save_path = saver.restore(self.sess, filename+".ckpt")
        print("Model restored")

    def get_variables_dict(self):
        return self.variables_dict

    def get_session(self):
        return self.sess

    def get_costlist(self):
        return self.costs_list

class CAE():


    def __init__(self,input_shape, row_bins, col_bins, n_filters, filter_sizes, strides, learning_rate=0.01,max_pool=False):


        n_features = row_bins*col_bins

        tf.reset_default_graph()

        # input to the network
        X = tf.placeholder( tf.float32, input_shape, name='x')

        X_tensor = tf.reshape(X, [-1, row_bins, col_bins, 1])

        current_input = X_tensor

        # notice instead of having 784 as our input features, we're going to have
        # just 1, corresponding to the number of channels in the image.
        # We're going to use convolution to find 16 filters, or 16 channels of information in each spatial location we perform convolution at.
        n_input = 1

        # We're going to keep every matrix we create so let's create a list to hold them all
        Ws = []
        shape = []
        layers_inputs = []

        # We'll create a for loop to create each layer:
        for layer_i, n_output in enumerate(n_filters):
            # just like in the last session,
            # we'll use a variable scope to help encapsulate our variables
            # This will simply prefix all the variables made in this scope
            # with the name we give it.
            with tf.variable_scope("encoder/layer/{}".format(layer_i)):
                # we'll keep track of the activations of each layer
                # As we'll need these for the decoder
                shape.append(current_input.get_shape().as_list())
                layers_inputs.append(current_input)
                # Create a weight matrix which will increasingly reduce
                # down the amount of information in the input by performing
                # a matrix multiplication
                W = tf.get_variable(
                    name='W',
                    shape=[
                        filter_sizes[layer_i],
                        filter_sizes[layer_i],
                        n_input,
                        n_output],
                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))#, regularizer = tf.contrib.layers.l2_regularizer(l2scale))


                # Now we'll convolve our input by our newly created W matrix
                # And then use a relu activation function on its output
                
                stri = strides[layer_i]
                
                if max_pool:
                    h = tf.nn.conv2d(current_input, W,strides=[1,1,1,1], padding='SAME')
                    h = tf.nn.relu(h)
                    h = tf.nn.max_pool(h, ksize=stri,strides=stri, padding='SAME')

                else:
                    h = tf.nn.conv2d(current_input, W,strides=stri, padding='SAME')
                    h = tf.nn.relu(h)
                
                current_input = h

                # Finally we'll store the weight matrix so we can build the decoder.
                Ws.append(W)

                # We'll also replace n_input with the current n_output, so that on the
                # next iteration, our new number inputs will be correct.
                n_input = n_output

        # Dropout?
        # keep_prob = tf.placeholder(tf.float32)
        # current_input = tf.nn.dropout(current_input, keep_prob)

        z = current_input

        print('Encoding')
        print('N Filters',n_filters)
        print('Filter sizes',filter_sizes)
        print('Shapes',shape)

        # %%
        # store the latent representation
        Ws.reverse()
        # and the shapes of each layer
        shape.reverse()
        # and the number of filters (which is the same but could have been different)
        n_filters.reverse()
        # and append the last filter size which is our input image's number of channels
        n_filters = n_filters[1:] + [1]

        print('Decoding')
        print('N Filters',n_filters)
        print('Filter sizes',filter_sizes)
        print('Shapes',shape)

        # %%
        # Build the decoder using the same weights
        for layer_i, shape in enumerate(shape):
            # we'll use a variable scope to help encapsulate our variables
            # This will simply prefix all the variables made in this scope
            # with the name we give it.
            with tf.variable_scope("decoder/layer/{}".format(layer_i)):
                layers_inputs.append(current_input)
                # Create a weight matrix which will increasingly reduce
                # down the amount of information in the input by performing
                # a matrix multiplication
                W = Ws[layer_i]

                stri = strides[len(shape)-layer_i-1]
                
                # Now we'll convolve by the transpose of our previous convolution tensor
                h = tf.nn.conv2d_transpose(current_input, W,
                    tf.stack([tf.shape(X)[0], shape[1], shape[2], shape[3]]),
                    strides=stri, padding='SAME')

                # And then use a relu activation function on its output
                current_input = tf.nn.relu(h)
                
        layers_inputs.append(current_input)

        # %%
        # now have the reconstruction through the network
        Y = current_input
        Y = tf.reshape(Y, [-1, n_features])

        # cost function measures pixel-wise difference

        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # reg_constant = 0.0  # Choose an appropriate one.
        # reg = reg_constant * tf.reduce_mean(reg_losses)

        reg = 0
        cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(X, Y), 1)) + reg

        self.variables_dict = {'X': X, 'z': z, 'Y': Y, 'cost': cost, 'layers_inputs':layers_inputs}

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)    

    def init_session(self):

        # We create a session to use the graph
        config = tf.ConfigProto( device_count = {'GPU': 0} )
        self.sess = tf.Session(config=config)
        self.sess.run(tf.initialize_all_variables())



    def train(self,data,batch_size,n_epochs,display_step=10 ):

        # %%
        # Fit all training data
        self.costs_list = []
        total_batch = int(data.length/batch_size)
        for epoch_i in range(n_epochs):
            for batch_i in range(total_batch):
                batch_xs = data.next_batch(batch_size)

                self.sess.run(self.optimizer, feed_dict={self.variables_dict['X']: batch_xs })
                cost_value = self.sess.run(self.variables_dict['cost'],  feed_dict={self.variables_dict['X']: batch_xs})
                self.costs_list.append(cost_value)
                    
            # Display logs per epoch step
            if epoch_i % display_step == 0:
                print("Epoch:", '%04d' % (epoch_i),"cost=", "{:.9f}".format(cost_value))

        return self.costs_list                

    def save(self,filename):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, filename+".ckpt")
        print("Model saved in file: %s" % save_path)

    def load(self,filename):
        saver = tf.train.Saver()        
        save_path = saver.restore(self.sess, filename+".ckpt")
        print("Model restored")

    def get_variables_dict(self):
        return self.variables_dict

    def get_session(self):
        return self.sess

    def get_costlist(self):
        return self.costs_list
