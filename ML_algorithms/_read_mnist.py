"""
http://yann.lecun.com/exdb/mnist/
"""

def read_minst_test():
  import os
  import struct
  import numpy as np
  # Load labels
  fname_labels = "data/t10k-labels-idx1-ubyte"
  flabels = open(fname_labels, 'rb')
  magic, num = struct.unpack(">II", flabels.read(8)) # read the beginning bytes
  labels = np.fromfile(flabels, dtype = np.int8)
  flabels.close()

  fname_images = "data/t10k-images-idx3-ubyte"
  fimages = open(fname_images, 'rb')
  magic, num, rows, cols = struct.unpack(">IIII", fimages.read(16)) # read the beginning bytes
  images = np.fromfile(fimages, dtype = np.uint8).reshape(len(labels), rows, cols)
  fimages.close()
  return images, labels

def show_image(image):
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  imgplot = ax.imshow(image, cmap = "Greys")
  ax.xaxis.set_ticks_position('top')
  ax.yaxis.set_ticks_position('left')
  plt.show()


images_test, labels_test = read_minst_test()
show_image(images_test[0])
