from conversion import convert,convert_mean_image
from models import *
import numpy as np
from PIL import Image

#lmodel = convert('vgg/proto','vgg/caffemodel')
#dump(lmodel,'model')

mean_image = convert_mean_image('vgg/meanfile')
np.save('vgg/mean_image',mean_image)
