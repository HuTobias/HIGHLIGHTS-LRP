import keras
import keras.backend as K
import tensorflow as tf


class ArgmaxPositions(keras.layers.Layer):
    ''' implementation of the LRP Argmax-Rule (https://link.springer.com/chapter/10.1007/978-3-030-30179-8_16)
        for convolutional layers

        Attributes
        ----------
        layer_output: the output of the layer to be analyzed
        layer_weights: the weights of the layer to be analyzed
        stride: the stride of the layer to be analyzed (has to be quadratic TODO make it free)
        filter_size: the filter size of the layer to be analyzed (has to be quadratic TODO make it free)
        relevance_values: the relevance values of the layer succeeding the layer to be analyzed


        Methods
        -------
        update_relevance:
            finds the most contributing position for a given relevance value and propagates the relevance to this position
        call:
            caluclates the relvance vlalues of the analyzed layer according to the argmax-rule.
            Assumes no padding.
     '''

    def __init__(self, stride, filter_size, layer_output, layer_weights, **kwargs):
        '''
        :param layer_output: the output of the layer to be analyzed
        :param layer_weights: the weights of the layer to be analyzed
        :param stride: the stride of the layer to be analyzed (has to be quadratic TODO make it free)
        :param filter_size: the filter size of the layer to be analyzed (has to be quadratic TODO make it free)
        '''
        super(ArgmaxPositions, self).__init__(**kwargs)
        self.stride = stride
        self.filter_size = filter_size
        self.layer_output = layer_output
        self.layer_weights = layer_weights

        # reference value for later
        self.zero = tf.constant(0, dtype=tf.float32)

    def build(self, input_shape):
        super(ArgmaxPositions, self).build(input_shape)  # Be sure to call this at the end

    def update_relevance(self, relevance_index):
        '''
        finds the most contributing position for a given relevance value and propagates the relevance to this position
        :param relevance_index: index of the relevance value to be analyzed
        :return: a mask with only one non-zero entry (with value equal to the relevance value to be analyzed)
                at the most contributing position. The mask has the same shape as the layer_output.
        '''
        # get x and y range of the relevant part of the picture
        x_start = self.stride * relevance_index[1]
        x_end = self.stride * relevance_index[1] + self.filter_size
        y_start = self.stride * relevance_index[2]
        y_end = self.stride * relevance_index[2] + self.filter_size

        # getting correct input in the batch
        batch_input = self.layer_output[relevance_index[0]]

        # getting the part/window of the input which corresponds to the relevance value
        input_patch = batch_input[x_start:x_end, y_start:y_end, :]

        # getting the filter which corresponds to the relevance value
        weight = self.layer_weights[:, :, :, relevance_index[3]]

        # callculating pointwise product, which is also used during forward propagation
        product = input_patch * weight

        # Getting local position of the most contributing neuron in this window
        old_shape = product.shape
        product = keras.layers.Flatten()(product)
        product = tf.expand_dims(product, axis=0)
        product = keras.layers.Flatten()(product)
        position = tf.argmax(product, axis=-1)[0]
        position = tf.unravel_index(position, old_shape)

        # getting global version of the local position
        global_position = [position[0] + x_start, position[1] + y_start, position[2]]

        # ravel gloabl position, indexes are shifted since out.shape has an empty first dimension
        global_index = global_position[0] * self.layer_output.shape[-2] * self.layer_output.shape[-1] + global_position[1] * self.layer_output.shape[-1] + \
                       global_position[2]

        # generate one_hot mask with the relevance value at the global argamx position
        mask = K.one_hot(global_index, tf.reduce_prod(batch_input.shape))
        relevance_value = self.relevance_values[0, relevance_index[1], relevance_index[2], relevance_index[3]]
        mask = mask * relevance_value
        mask = K.reshape(mask, batch_input.shape)

        return mask

    def call(self, inputs):
        '''
            Caluclates the relvance vlalues of the analyzed layer according to the argmax-rule.
            Assumes no padding.
            :param inputs: the relevance values of the layer succeeding the layer to be analyzed
            :returns: an array containing the relvance vlalues of the analyzed layer according to the argmax-rule
        '''
        # get all non-zero relevance values
        self.relevance_values = inputs
        where = tf.not_equal(inputs, self.zero)
        indices = tf.where(where)

        # generate a mask for each relevance value, which only has this relevance value as non-zero value
        # at the most contributing position
        relevance_masks = tf.map_fn(self.update_relevance, indices, dtype=tf.float32)

        #sum over all masks
        new_relevance_array = tf.reduce_sum(relevance_masks, axis=0)

        #this is needed for shaping reasons TODO innvestigate further
        new_relevance_array = K.expand_dims(new_relevance_array, axis=0)

        return new_relevance_array