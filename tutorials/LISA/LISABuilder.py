import PIL
import random
import pathlib
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

class LISABuilder(tfds.core.GeneratorBasedBuilder):
  """
    Dataset builder for LISA Traffic dataset.
  """
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.'
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """
      Dataset metadata:
      Homepage: https://cvrr.ucsd.edu/lisa-traffic-signs-dataset
      Citation: Andreas Møgelmose, Mohan M. Trivedi, and Thomas B. Moeslund,
        “Vision based Traffic Sign Detection and Analysis for Intelligent Driver Assistance Systems: Perspectives and Survey,”
        IEEE Transactions on Intelligent Transportation Systems, 2012.
    """
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(960, 1280, 3)),
            'bounding boxes': tfds.features.Sequence({
                'bbox': tfds.features.BBoxFeature(doc="[yxyx] with y / height, x / width"),
                'label': tfds.features.ClassLabel(num_classes=7)
            })
        }),
        disable_shuffling=False
    )

  def train_test_split(self, img_labels_dict, training_percent = 0.80):
    # Shuffle the dictionary by the keys
    keys = list(img_labels_dict.keys())
    random.shuffle(keys)

    nkeys = int(len(keys) * training_percent)
    training_keys = keys[:nkeys]
    test_keys = keys[nkeys:]

    training_dict = {k: img_labels_dict[k] for k in training_keys}
    test_dict = {k: img_labels_dict[k] for k in test_keys}

    return training_dict, test_dict

  def format_bounding_boxes(self, upper_left_x, upper_left_y, 
                            lower_right_x, lower_right_y, 
                            height, width):
    """
      Package the coordinates into a tf.Tensor.
    """
    # Return ymin, xmin, ymax, xmax - yxyx format divided by height and width
    ymin = upper_left_y / height
    xmin = upper_left_x / width
    ymax = lower_right_y / height
    xmax = lower_right_x / width
    return np.asarray([ymin, xmin, ymax, xmax])

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    annot_path = pathlib.Path('./LISA_processed/annotations/')
    img_path = pathlib.Path('./LISA_processed/images/')
    classes = ['stop', 'go', 'warning', 'warningLeft', 'stopLeft', 'goForward', 'goLeft']
    classes = sorted(classes)
    class_mapping = dict((name, idx) for idx, name in enumerate(classes))

    data_dict = dict() # Let each image have their entries for the bbox coordinates and classes
    for img_file in img_path.glob("*.jpg"):
      for annot_file in annot_path.glob("*.csv"):
        # Turn the .csv file into a dataframe for easy reading
        df = pd.read_csv(str(annot_file), delimiter=',')
        if any(img_file.name == name for name in df['Filename'].unique()):
          # Get indices of matched up file names 
          idx = df.index[df['Filename'] == img_file.name].tolist()
          upper_left_x, upper_left_y, lower_right_x, lower_right_y, classes = [], [], [], [], []
          for i in idx:
            # Get the bounding box information: upper_left_x, upper_left_y,	lower_right_x	, lower_right_y
            upper_left_x.append(df.loc[i, 'Upper left corner X']) # xmin
            upper_left_y.append(df.loc[i, 'Upper left corner Y']) # ymin
            lower_right_x.append(df.loc[i, 'Lower right corner X']) # xmax
            lower_right_y.append(df.loc[i, 'Lower right corner Y']) # ymax
            # Get the class information
            classes.append(df.loc[i, 'Annotation tag'])
          # Ensure all new lists are the same length - sanity check
          assert len(upper_left_x) == len(upper_left_y) == len(lower_right_x) == len(lower_right_y) == len(classes)
          data_dict[img_file.name] = {'xmin': upper_left_x,
                                      'ymin': upper_left_y,
                                      'xmax': lower_right_x,
                                      'ymax': lower_right_y,
                                      'classes': classes}

    training_dict, test_dict = self.train_test_split(data_dict)

    return {
        'train': self._generate_examples(training_dict, class_mapping),
        'test': self._generate_examples(test_dict, class_mapping)
    }

  def _generate_examples(self, data_dict, class_mapping):
    image_id = 0
    for key in data_dict:
      image = PIL.Image.open('./LISA_processed/images/' + key)
      image_tensor = np.asarray(image)
      image_id += 1

      height, width, _ = image_tensor.shape

      bbox_coords, cls = [], []
      # Recall the value of this dictionary is also a dictionary
      for i in range(len(data_dict[key]['xmin'])): # All lists are same length, does not matter which one you choose
        # Get each individual coordinate and corresponding class
        xmin = data_dict[key]['xmin'][i]
        ymin = data_dict[key]['ymin'][i]
        xmax = data_dict[key]['xmax'][i]
        ymax = data_dict[key]['ymax'][i]
        # Format the bounding box coordinates
        bbox_coords.append(self.format_bounding_boxes(xmin, ymin, 
                                                      xmax, ymax, 
                                                      height, width))
        cls.append(class_mapping[data_dict[key]['classes'][i]])

      example = {'image': image_tensor,
                 'bounding boxes': {
                     'bbox': bbox_coords,
                     'label': cls
                 }
                }

      yield image_id, example  
