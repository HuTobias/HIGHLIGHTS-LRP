import keras
import keras.backend as K
from innvestigate.analyzer.relevance_based.relevance_rule import Alpha1Beta0IgnoreBiasRule
from innvestigate.analyzer.relevance_based.relevance_analyzer import _LRPFixedParams
import innvestigate
import custom_layers


class Argmax(_LRPFixedParams):
    """LRP-analyzer that uses the argmax rule"""

    def __init__(self, model, *args, **kwargs):
        innvestigate.analyzer.relevance_based.relevance_analyzer.LRP_RULES['Argmax'] = ArgmaxRule
        super(Argmax, self).__init__(model, *args, rule="Argmax", **kwargs)


class ArgmaxRule(Alpha1Beta0IgnoreBiasRule):
    '''
        implementation of the ArgmaxRule described in https://link.springer.com/chapter/10.1007/978-3-030-30179-8_16
    '''
    def __init__(self, *args, **kwargs):
        layer = args[0]
        self.layer = layer

        # use classical LRP rule for non-convolutional layers and first convolutional layer after the input
        if layer.__class__ == keras.layers.convolutional.Conv2D:
            if layer._inbound_nodes[0].inbound_layers[0].__class__ == keras.engine.input_layer.InputLayer:
                super(Alpha1Beta0IgnoreBiasRule, self).__init__(*args,
                                                                alpha=1,
                                                                beta=0,
                                                                bias=False,
                                                                **kwargs)
            else:
                self.weights = layer.weights[0]
                self.padding = layer.padding
                self.filter_size = layer.kernel_size[0]
                #TODO assumes quadratic stride for now
                self.stride = layer.strides[0]
        else:
            super(Alpha1Beta0IgnoreBiasRule, self).__init__(*args,
                                                                     alpha=1,
                                                                     beta=0,
                                                                     bias=False,
                                                                     **kwargs)


    def apply(self, Xs, Ys, Rs, reverse_state):
        # use classical LRP rule for non-convolutional layers and first convolutional layer after the input
        if self.layer.__class__ != keras.layers.convolutional.Conv2D:
            return super(Alpha1Beta0IgnoreBiasRule, self).apply(Xs,Ys,Rs,reverse_state)
        if self.layer._inbound_nodes[0].inbound_layers[0].__class__ == keras.engine.input_layer.InputLayer:
            return super(Alpha1Beta0IgnoreBiasRule, self).apply(Xs, Ys, Rs, reverse_state)

        #remove additional array from innvestigate LRP stuff
        #TODO this could break something
        Xs = Xs[0] #TODO make this more generall
        Rs = Rs[0]

        weights = self.weights
        padding = self.padding
        filter_size = self.filter_size
        stride = self.stride
        input_shape = Xs.get_shape().as_list()[1:]

        # getting the output of the previous layer and add padding if it was used while training the net
        if (padding != 0):
            if (padding == "valid"):
                #TODO non quadratic filters
                filter_height = filter_size
                filter_width = filter_size

                # valid padding doesnt really pad but only considers the values outside the original input as zero, so we just pad enough zeros at the ends of the input
                pad_top = 0
                pad_bottom = filter_height
                pad_left = 0
                pad_right = filter_size

                # padded_input
                out = K.spatial_2d_padding(Xs, padding=((pad_top, pad_bottom), (pad_left, pad_right)))

            elif (padding == "same"):
                in_height = input_shape[0]
                in_width = input_shape[1]
                filter_height = filter_size
                filter_width = filter_size
                strides = (1, stride, stride, 1)

                # calculate padding according to https://www.tensorflow.org/api_guides/python/nn#Convolution ( tf version 1.11 )
                if (in_height % strides[1] == 0):
                    pad_along_height = max(filter_height - strides[1], 0)
                else:
                    pad_along_height = max(filter_height - (in_height % strides[1]), 0)
                if (in_width % strides[2] == 0):
                    pad_along_width = max(filter_width - strides[2], 0)
                else:
                    pad_along_width = max(filter_width - (in_width % strides[2]), 0)

                pad_top = pad_along_height // 2
                pad_bottom = pad_along_height - pad_top
                pad_left = pad_along_width // 2
                pad_right = pad_along_width - pad_left

                #padded_input
                out = K.spatial_2d_padding(Xs, padding=((pad_top, pad_bottom), (pad_left, pad_right)))

            else:
                raise ValueError('as of now only *same*,*valid* and *0* are supported paddings')

            self.input_vector_length = 1
            shape = list(out.shape)
            shape.pop(0)
            for i in shape:
                self.input_vector_length *= i

        # the actual argmax part
        new_relevance_array = custom_layers.ArgmaxPositions(stride, filter_size, out, weights)(Rs)

        #remove padding, if it was added
        if (padding != 0):
            new_relevance_array =  keras.layers.Lambda( lambda x: x[:,pad_top:-pad_bottom, pad_left:-pad_right, :])(new_relevance_array)

        return new_relevance_array





