
#import matplotlib.pyplot as plt
from keras import *
from keras import backend as K
import numpy as np
from PIL import Image

## some DNN model has an explicit input layer
def is_input_layer(layer):
  return str(layer).find('InputLayer')>=0

def is_conv_layer(layer):
  return str(layer).find('conv')>=0 or str(layer).find('Conv')>=0

def is_dense_layer(layer):
  return str(layer).find('dense')>=0 or str(layer).find('Dense')>=0

def is_activation_layer(layer):
  return str(layer).find('activation')>=0 or str(layer).find('Activation')>=0

def get_activation(layer):
  if str(layer.activation).find('relu')>=0: return 'relu'
  elif  str(layer.activation).find('linear')>=0: return 'linear'
  elif  str(layer.activation).find('softmax')>=0: return 'softmax'
  else: return ''

def is_maxpooling_layer(layer):
  return str(layer).find('MaxPooling')>=0 

def is_flatten_layer(layer):
  return str(layer).find('flatten')>=0 or str(layer).find('Flatten')>=0

def is_dropout_layer(layer):
  return False ## we do not allow dropout

class cover_layert:
  def __init__(self, layer, layer_index, is_conv):
    self.layer=layer
    self.layer_index=layer_index
    self.is_conv=is_conv
    self.activations=[]
    self.pfactor=1.0 ## the proportional factor
    self.actiavtions=[] ## so, we need to store neuron activations?
    self.nc_map=None ## to count the coverage

  def initialize_nc_map(self):
    sp=self.layer.output.shape
    if self.is_conv:
      self.cover_map = np.zeros((1, sp[1], sp[2], sp[3]), dtype=bool)
    else:
      self.cover_map = np.zeros((1, sp[1]), dtype=bool)

  def update_activations(self):
    self.activations=[np.multiply(act, self.nc_map) for act in self.activations]
    #for i in range(0, len(self.activations)):
    #  self.activations[i] = np.multiply(self.activations[i], self.nc_map)

def get_layer_functions(dnn):
  layer_functions=[]
  for l in range(0, len(dnn.layers)):
    layer=dnn.layers[l]
    current_layer_function=K.function([layer.input], [layer.output])
    layer_functions.append(current_layer_function)
  return layer_functions

def get_cover_layers(dnn, criterion):
  cover_layers=[]
  for l in range(0, len(dnn.layers)):
    layer=dnn.layers[l]
    if is_conv_layer(layer) or is_dense_layer(layer):
      sp=layer.output.shape
      clayer=cover_layert(layer, l, is_conv_layer(layer))
      cover_layers.append(clayer)
  return cover_layers


### given an input image, to evaluate activations
def eval(layer_functions, im):
  activations=[]
  for l in range(0, len(layer_functions)):
    if l==0:
      activations.append(layer_functions[l]([[im]])[0])
    else:
      activations.append(layer_functions[l]([activations[l-1]])[0])
  return activations

def eval_batch(layer_functions, ims):
  activations=[]
  for l in range(0, len(layer_functions)):
    if l==0:
      activations.append(layer_functions[l]([ims])[0])
    else:
      activations.append(layer_functions[l]([activations[l-1]])[0])
  return activations

class raw_datat:
  def __init__(self, data, labels):
    self.data=data
    self.labels=labels


class test_objectt:
  def __init__(self, dnn, raw_data, criterion, norm):
    self.dnn=dnn
    self.raw_data=raw_data
    ## test config
    self.norm=norm
    self.criterion=criterion
    self.channels_last=True

def calculate_pfactors(activations, cover_layers):
  fks=[]
  for clayer in cover_layers:
    layer_index=clayer.layer_index
    sub_acts=np.abs(activations[layer_index])
    fks.append(np.average(sub_acts))
  av=np.average(fks)
  for i in range(0, len(fks)):
    cover_layers[i].pfactor=av/fks[i]

def save_an_image(im, title, di='./'):
  ## we assume im is normalized
  img=Image.fromarray(np.uint8(im*255))
  img.save(di+title+'.png')

#def show_adversarial_examples(imgs, ys, name):
#  for i in range(0, 2):
#    plt.subplot(1, 2, 1+i)
#    print 'imgs[i].shape is ', imgs[i].shape
#    plt.imshow(imgs[i].reshape([28,28]), cmap=plt.get_cmap('gray'))
#    plt.title("label: "+str(ys[i]))
#    plt.savefig(name, bbox_inches='tight')
