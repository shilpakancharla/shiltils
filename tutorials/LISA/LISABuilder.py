import PIL
import random
import pathlib
import numpy as np
import pandas as pd

import keras
import keras_cv
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

  def format_bounding_boxes(self, upper_left_x, upper_left_y, lower_right_x, lower_right_y):
    """
      Package the coordinates into a tf.Tensor.
    """
    # Return ymin, xmin, ymax, xmax - yxyx format
    return np.asarray([upper_left_y, upper_left_x, lower_right_y, lower_right_x])
  
  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    download_annot_dir = pathlib.Path('./LISA/annotations/')
    download_img_dir = pathlib.Path('./LISA/images/')
    classes = ['stop', 'go', 'warning', 'warningLeft', 'stopLeft', 'goForward', 'goLeft']
    classes = sorted(classes)
    class_mapping = dict((name, idx) for idx, name in enumerate(classes))

    image_labels_dict = dict()
    for f in download_img_dir.rglob("*/*.jpg"):
      for a in download_annot_dir.rglob("*/*.csv"):
        if str(a).split('/')[-2] in str(f):
          image_labels_dict[str(f)] = str(a)

    training_dict, test_dict = self.train_test_split(image_labels_dict)

    return {
        'train': self._generate_examples(training_dict, # Specify training dictionary
                                          class_mapping), 
        'test': self._generate_examples(test_dict, # Specify test dictionary
                                        class_mapping) 
    }

  def _generate_examples(self, data_dictionary, class_mapping):
    image_id = 0
    for key in data_dictionary:
      image = PIL.Image.open(key)
      image_tensor = np.asarray(image)
      image_id += 1

      height, width, _ = image_tensor.shape

      # Drop unecessary rows from dataframe, this will save you some memory
      df = pd.read_csv(data_dictionary[key],
                       delimiter=';').drop(['Origin file',
                                            'Origin frame number',
                                            'Origin track',
                                            'Origin track frame number'], axis=1)

      # Get the dataframe row that has the 'Filename' as the key
      row = df.loc[df['Filename'].str.contains(key.split('/')[-1])]

      if len(row) > 0:
        cls, bbox_coords = [], []
        for idx, row in row.iterrows(): # Iterate over the rows of the dataframe
          class_ = row['Annotation tag']
          upper_left_x = row['Upper left corner X'] # xmin
          upper_left_y = row['Upper left corner Y'] # ymin
          lower_right_x = row['Lower right corner X'] # xmax
          lower_right_y = row['Lower right corner Y'] # ymax
          cls.append(class_mapping[class_])
          bbox_coords.append(self.format_bounding_boxes(upper_left_x / width,
                                                        upper_left_y / height,
                                                        lower_right_x / width,
                                                        lower_right_y / height))
        example = {'image': image_tensor,
                   'bounding boxes': {
                     'bbox': bbox_coords,
                     'label': cls
                     }
                  }

        yield image_id, example
