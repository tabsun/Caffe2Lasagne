'''
defines some extra layers in lasagne form to use to convert caffe models
'''

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from lasagne.layers import cuda_convnet


import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities
import lasagne.init as init

from theano.sandbox.cuda.basic_ops import gpu_contiguous
# need to do my change to the following:
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs


class CaffeConv2DCCLayer(cuda_convnet.Conv2DCCLayer):
    def __init__(self, incoming, num_filters, filter_size, groups=1, stride=(1, 1), border_mode=None, untie_biases=False, W=init.Uniform(), b=init.Constant(0.), nonlinearity=nonlinearities.rectify, pad=None, dimshuffle=True, flip_filters=False, partial_sum=1, **kwargs):
        super(CaffeConv2DCCLayer,self).__init__(incoming, num_filters, filter_size, stride=stride, untie_biases=untie_biases, W=W, b=b, nonlinearity=nonlinearity, pad=pad, dimshuffle=dimshuffle, flip_filters=flip_filters, partial_sum=partial_sum, **kwargs)
        self.groups = groups
        # the FilterActs in pylearn2 cannot accept tuple-type pad
        if isinstance(self.pad, int):
            self.pad = self.pad
        elif isinstance(self.pad, tuple):
            self.pad = self.pad[0]
        else:
            self.pad = 0
        self.filter_acts_op = FilterActs(numGroups=self.groups,
                                         stride=self.stride,
                                         partial_sum=self.partial_sum,
                                         pad=self.pad)


class CaffeMaxPool2DCCLayer(cuda_convnet.MaxPool2DCCLayer):
    def __init__(self,**kwargs):
        raise NotImplementedError

        
