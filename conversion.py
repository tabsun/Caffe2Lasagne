import theano
import numpy as np
import time
import theano.tensor as T
from caffe_layers import extra_layers
from models import *
# see if GPU + pylearn2 is available for the cuda convnet wrappers
# I encounter compile error when using pylearn2 so I removed cuda_convnet usage though GPU+pylearn2 are available
##try:
##	from  caffe_layers import extra_convnet_layers
##	from lasagne.layers import cuda_convnet
##	print "===============\n"*5
##	print "using cuda convnet wrappers"
##	print "===============\n"*5
##	cuda = True
##except Exception as e:
##	print e
print "probably no GPU, or no pylearn2 capabilities; using normal"
cuda = False
import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities

# see if caffe is available
try:
	from parsing import parse_from_protobuf_caffe
	caffe_parsing = True
except ImportError as e:
	print "no caffe found; parsing will be done from google protobuf (slower)"
        from parsing import parse_from_protobuf
	caffe_parsing=False
parse_model_def = parse_from_protobuf.parse_model_def



V1Map = {0: 'NONE',
 1: 'ACCURACY',
 2: 'BNLL',
 3: 'CONCAT',
 4: 'CONVOLUTION',
 5: 'DATA',
 6: 'DROPOUT',
 7: 'EUCLIDEAN_LOSS',
 8: 'FLATTEN',
 9: 'HDF5_DATA',
 10: 'HDF5_OUTPUT',
 11: 'IM2COL',
 12: 'IMAGE_DATA',
 13: 'INFOGAIN_LOSS',
 14: 'INNER_PRODUCT',
 15: 'LRN',
 16: 'MULTINOMIAL_LOGISTIC_LOSS',
 17: 'POOLING',
 18: 'RELU',
 19: 'SIGMOID',
 20: 'SOFTMAX',
 21: 'SOFTMAX_LOSS',
 22: 'SPLIT',
 23: 'TANH',
 24: 'WINDOW_DATA',
 25: 'ELTWISE',
 26: 'POWER',
 27: 'SIGMOID_CROSS_ENTROPY_LOSS',
 28: 'HINGE_LOSS',
 29: 'MEMORY_DATA',
 30: 'ARGMAX',
 31: 'THRESHOLD',
 32: 'DUMMY_DATA',
 33: 'SLICE',
 34: 'MVN',
 35: 'ABSVAL',
 36: 'SILENCE',
 37: 'CONTRASTIVE_LOSS',
 38: 'EXP',
 39: 'DECONVOLUTION'}

# valid names (lowercased)
valid_convolution = set(['convolution'])
valid_inner_product = set(['innerproduct', 'inner_product'])
valid_relu = set(['relu'])
valid_lrn = set(['lrn'])
valid_pooling = set(['pooling'])
valid_softmax = set(['softmax'])
valid_dropout = set(['dropout'])

# ======= [ FUNCTIONS THAT YOU MIGHT USE: ] ========= #
def convert_trained_caffemodel(lasagne_model, caffemodel, prototxt='', caffe_parse = caffe_parsing):
	'''
	PARAMETERS:
		- lasagne_model: A lasagne model (i.e, BaseModel) to set parameters of. This probably came from a call to
		convert_model_def. The layers should be named the same as the caffemodel.
		- caffemodel: /path/to/trained_model.caffemodel file
		- prototxt: /path/to/deploy.prototxt file. This is not needed if you don't use the caffe parsing, otherwise it's needed.
		- caffe_parse: don't worry about this, it will be set based on whether or not you have caffe installed.

	NOTES:
	sets the params of lasagne_model to be from the trained caffemodel.
	'''
	# load in the caffemodel (this takes a long time without cpp implementation of protobuf)
	if caffe_parse:
		layer_params = parse_from_protobuf_caffe.parse_caffemodel(caffemodel, prototxt=prototxt) # prototxt is passed in if caffemodel uses caffe
	else:
		layer_params = parse_from_protobuf.parse_caffemodel(caffemodel)

	# this should be in the same order as was made by the lasagne model, but reversed. we will check that.
	# todo: maybe just go by names, strictly?
	for lasagne_layer in lasagne_model.all_layers:
		if len(lasagne_layer.get_params()) == 0:
			# no params to set
			continue
		print lasagne_layer.name
		lp = layer_params[lasagne_layer.name]
		Wblob = lp[0]
		bblob = lp[1]
		# get arrays of parameters		
		W = array_from_blob(Wblob)
		b = array_from_blob(bblob)
		print "W: ", W.shape
		print "b: ", b.shape
		# set parameters
		set_model_params(lasagne_layer, W,b)

def convert_model_def(prototxt):
	'''
	PARAMETERS:
		- prototxt: /path/to/deploy.prototxt file

	NOTES:
	returns a lasagne model with the correct layers
	'''
	name, inp, architecture = parse_model_def(prototxt)
	inp_name, inp_dims = inp

        input_T = T.tensor4("Input")
	last_layer = inp_layer = layers.InputLayer(shape=[1,inp_dims[1],
                                                inp_dims[2],inp_dims[3]],
                                                   name=inp_name,input_var=input_T)
	print last_layer.name
	print [None,inp_dims[1],inp_dims[2],inp_dims[3]]
	# now go through all layers and create the lasagne equivalent
	for layer in architecture:
		this_layer = parse_layer_from_param(layer, last_layer)
		last_layer = this_layer

        print "begin compiling"
	# make the lasagne model (only need last layer)
	model = BaseModel(last_layer)
	print "compiling done"
	return model


def convert(prototxt, caffemodel, caffe_parse=caffe_parsing):
	'''
	PARAMETERS:
		-prototxt: /path/to/deploy.prototxt file
		-caffemodel: /path/to/trained_model.caffemodel file
		-caffe_parse: don't worry about this

	NOTES:
	wraps the two methods above to return a lasagne model with the correct parameters.
	'''
        print "begin to convert prototxt"
	lmodel = convert_model_def(prototxt)
	print "Done conversion of prototxt"
	convert_trained_caffemodel(lmodel, caffemodel, prototxt,caffe_parse=caffe_parse)
	return lmodel

def convert_mean_image(binaryproto):
	'''
	PARAMETERS:
		-binaryproto: /path/to/mean_iamge.binaryproto

	NOTES:
	returns a numpy array of the mean image given by the binaryproto file.
	'''
	return parse_from_protobuf.parse_mean_file(binaryproto)




# ===== [ HELPERS ] ===== #

def parse_layer_from_param(layer,last_layer):
	'''
	returns the correct layer given the param dict
	'''
	if type(layer.type) == int:
		# this is the legacy caffe thing
		tp = V1Map[layer.type].lower()
	else:
		tp = layer.type.lower()

	if tp in valid_convolution:
		if cuda==True:
			return cuda_conv_layer(layer, last_layer)
		else:
			return conv_layer(layer, last_layer)
	elif tp in valid_relu:
		return relu_layer(layer, last_layer)
	elif tp in valid_pooling:
		if cuda==True:
			return cuda_pooling_layer(layer, last_layer)
		else:
			return pooling_layer(layer, last_layer)
	elif tp in valid_inner_product:
		return ip_layer(layer, last_layer)
	elif tp in valid_dropout:
		return dropout_layer(layer, last_layer)
	elif tp in valid_softmax:
		return softmax_layer(layer, last_layer)
	elif tp in valid_lrn:
		return lrn_layer(layer, last_layer)
	else:
		raise Exception('not a valid layer: %s' % tp)



def lrn_layer(layer, last_layer):
	name = layer.name
	param = layer.lrn_param
	# set params
	alpha = param.alpha
	beta = param.beta
	n = param.local_size

	lrn = extra_layers.CaffeLocalResponseNormalization2DLayer(last_layer, alpha=alpha, beta=beta, n=n,name=name)
	return lrn

def cuda_conv_layer(layer, last_layer):
	name = layer.name
	param = layer.convolution_param

	num_filters = param.num_output
	filter_size = (param.kernel_size,param.kernel_size) #only suppose square filters
	stride = (param.stride,param.stride) # can only suport square strides anyways
	## border mode is wierd...
	border_mode = None
	pad = param.pad
	nonlinearity=nonlinearities.identity
	groups= param.group
			
	conv = extra_convnet_layers.CaffeConv2DCCLayer(last_layer, groups=groups, num_filters=num_filters,filter_size=filter_size, stride=stride, border_mode=border_mode, pad=pad, nonlinearity=nonlinearity,name=name)
	return conv

def conv_layer(layer, last_layer):
	name = layer.name
	print name
	param = layer.convolution_param
	num_filters = param.num_output
	filter_size = (param.kernel_size, param.kernel_size)
	stride = (param.stride, param.stride)
	group = param.group
	pad = param.pad
	nonlinearity=nonlinearities.identity

	# theano's conv only allows for certain padding, not arbitrary. not sure how it will work if same border mode is not true.
	if (filter_size[0] - pad * 2 ) == 1:
		print "using same convolutions, this should be correct"
		border_mode = 'same'
	elif pad == 0:
		print "using valid border mode, this should work but who knows"
		border_mode='valid'
	elif pad != 0:
		print "pretty sure this won't work but we'll try a full conv"
		border_mode = 'full'
	if group > 1:
		conv = extra_layers.CaffeConv2DLayer(last_layer, group=group,num_filters=num_filters, filter_size=filter_size, stride=stride, border_mode=border_mode, nonlinearity=nonlinearity,name=name)
	else:
		conv = layers.Conv2DLayer(last_layer, num_filters=num_filters, filter_size=filter_size, pad=border_mode, stride=stride, nonlinearity=nonlinearity,name=name)
	return conv

def relu_layer(layer, last_layer):
	name = layer.name
	print name
	return extra_layers.ReluLayer(last_layer,name=name)

def pooling_layer(layer, last_layer):
	name = layer.name
	print name
	param = layer.pooling_param
	pool_size=(param.kernel_size,param.kernel_size) #caffe only does square kernels
	stride = (param.stride, param.stride)
	if stride[0] != pool_size[0]:
		pool = extra_layers.CaffeMaxPool2DLayer(last_layer,pool_size, stride=stride,name=name)
	else:
		pool = layers.MaxPool2DLayer(last_layer, pool_size=pool_size,stride=stride,name=name) # ignore border is set to False, maybe look into how caffe does borders if the strides don't work perfectly
	return pool

def cuda_pooling_layer(layer, last_layer):
	name = layer.name
	print name
	param = layer.pooling_param
	kernel_size=param.kernel_size
	stride = (param.stride, param.stride)

	pool = cuda_convnet.MaxPool2DCCLayer(last_layer, kernel_size, stride=stride,name=name)
	return pool

def ip_layer(layer, last_layer):
	name = layer.name
	param = layer.inner_product_param
	num_units=param.num_output
	nonlinearity=nonlinearities.identity

	dense = layers.DenseLayer(last_layer, num_units=num_units, nonlinearity=nonlinearity,name=name)
	return dense

def dropout_layer(layer, last_layer):
	name = layer.name
	print name
	'''
	TODO: IMPLEMENT THIS. currently only using this script for forward passes, so this can be a complete identity
	but in the future maybe i'll want to finetune, so this would need to be implemented.
	'''
	return extra_layers.IdentityLayer(last_layer, name=name)

def softmax_layer(layer, last_layer):
	name = layer.name
	print name
	return extra_layers.SoftmaxLayer(last_layer, name=name)

def set_model_params(lasagne_layer,W,b):
	if cuda:
		if isinstance(lasagne_layer, cuda_convnet.Conv2DCCLayer):
			set_cuda_conv_params(lasagne_layer, W,b)
			return

	if isinstance(lasagne_layer, layers.Conv2DLayer):
		set_conv_params(lasagne_layer,W,b)
		return
	elif isinstance(lasagne_layer, layers.DenseLayer):
		set_ip_params(lasagne_layer, W,b)
		return
	else:
		raise Exception ("don't know this layers: %s" % type(lasagne_layer))



def array_from_blob(blob):
        if blob.num == 0 or blob.channels == 0 or blob.height == 0 or blob.width == 0:
                norm_shape = ()
                for dim in blob.shape.dim:
                        norm_shape = norm_shape+(int(dim),)  
        else:
                norm_shape = (blob.num,blob.channels,blob.height,blob.width)
        # if you use blob.shape to get size, it may be 1-dim tupel so need to
        # extend by adding (1,1....)
        while len(norm_shape) < 4:
                norm_shape = (1,) + norm_shape
        return np.array(blob.data).reshape(norm_shape)

def set_conv_params(layer, W,b):
	# b needs to just be the last index
	b = b[0,0,0,:]
	# W needs to be fixed
	W = W[:,:,::-1,::-1]
	layer.W.set_value(W.astype(theano.config.floatX))
	layer.b.set_value(b.astype(theano.config.floatX))

def set_cuda_conv_params(layer,W,b):
	# b needs to just be the last index
	b = b[0,0,0,:]
	# W needs to be reshaped into n_features(from prev layer), size, size, n_filters
	layer.W.set_value(W.astype(theano.config.floatX))
	layer.b.set_value(b.astype(theano.config.floatX))

def set_ip_params(layer,W,b):
	# W needs to just be the last 2, shuffled
	W = W[0,0,:,:].T
	# b needs to just be the last index
	b = b[0,0,0,:]
	layer.W.set_value(W.astype(theano.config.floatX))
	layer.b.set_value(b.astype(theano.config.floatX))


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--prototxt", default='VGG_ILSVRC_16_layers_deploy.prototxt', help="model definition file")
	parser.add_argument("--caffemodel", default='VGG_ILSVRC_16_layers.caffemodel',help="model binary")
	args = parser.parse_args()

	model, net = convert(args.prototxt,args.caffemodel)
	print 'testing similarity...'
	random_mat, outlist =test_similarity(model, net)
	test_serialization(model, random_mat)
