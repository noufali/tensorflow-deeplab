import os
from io import BytesIO
import tarfile
import tempfile
import scipy.misc
from option import Options
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image,name):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    ### SEGMENTATION MAP ###
    seg_map = batch_seg_map[0]  
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    array_resized = scipy.misc.imresize(seg_image,(height,width),interp='nearest',mode=None)
    # save
    path = '/home/paperspace/tensorflow-deeplab/images/'
    scipy.misc.imsave(path + 'seg.jpg',array_resized)
    
    ### MAKE PEOPLE TRANSPARENT ###
    seg_saved = Image.open(path + 'seg.jpg').convert("RGBA")
    width, height = seg_saved.size
    def almostEquals(a,b,thres=30):
      return all(abs(a[i]-b[i])<thres for i in range(len(a)))
    # if you find pixels that are almost black then make them white, otherwie make transparent
    for x in range(width):
      for y in range(height):
        current_color = seg_saved.getpixel((x,y))
        if almostEquals(current_color,(0,0,0,255)):
          seg_saved.putpixel((x,y),(255,255,255,255))
        else:
          seg_saved.putpixel((x,y),(255,255,255,0))
    seg_saved.save(path + "seg_changed.png")
    
    ### APPLY NEW SEGMENTATION MAP ON ORIGINAL IMAGE AND FLIP IMAGE ###
    seg_2 = Image.open(path + "seg_changed.png").convert("RGBA")
    original = image.convert("RGBA")
    original.paste(seg_2,(0,0),seg_2)
    rotated_image = original.transpose(Image.FLIP_LEFT_RIGHT)
    # remove pixels surrounding person
    width, height = rotated_image.size
    for x in range(width):
      for y in range(height):
        current_color = rotated_image.getpixel((x,y))
        if almostEquals(current_color,(255,255,255,255),5):
          rotated_image.putpixel((x,y),(255,255,255,0))
    rotated_image.save(path + name + "_mask.png")
    # remove files
    #os.remove(path + "seg.jpg")
    #os.remove(path + "seg_changed.png")

    #style = Image.open(path + "1.jpg").convert("RGBA")
    #man = Image.open(path + "farmer_mask.png").convert("RGBA")
    #style.paste(man,(0,0),man)
    #style.save(path + name + "_last.png")
    #Image.blend(style,man,.7).save(path + name + "_last.png")
    
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


LABEL_NAMES = np.asarray([
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

path = "/home/paperspace/tensorflow-deeplab/datasets/deeplab_model.tar.gz"
MODEL = DeepLabModel(path)
print('model loaded successfully!')


def run_visualization(args):
  """Inferences DeepLab model and visualizes result."""
  image = args.image
  try:
    original_im = Image.open(image)
    filename = os.path.splitext(os.path.basename(image))[0]
  except IOError:
    print('Cannot retrieve image. Please check url: ' + image)
    return

  print('running deeplab on image %s...' % image)
  resized_im, seg_map = MODEL.run(original_im,filename)

  #vis_segmentation(resized_im, seg_map)


#image_path = "/home/paperspace/tensorflow-deeplab/images/farmer.jpg"
#run_visualization(image_path)

def main():
	# getting things ready
	args = Options().parse()
	if args.subcommand is None:
		raise ValueError("ERROR: specify the image")
	# run demo
	run_visualization(args)

if __name__ == '__main__':
	main()



