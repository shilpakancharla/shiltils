import pandas as pd
import tensorflow as tf

class FrameGenerator:
  def __init__(self, dataset, batch_size, num_classes):
    """
      Initialize the generator class to create a TensorFlow dataset.

      Args:
        dataset: Dataframe containing the filtered frames and encoded labels from preprocessing.
        batch_size: Batch size to feed into the model.
        num_classes: Number of categories to classify.
    """
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_classes = num_classes

  def __call__(self):
    """
      Generator that will result in a batch of frames and labels to be fed into the model.
    """
    for i in range(0, len(self.dataset), self.batch_size):
      batch_samples = self.dataset[i:i + self.batch_size]
      stacked_tensor_frames_list = []
      label_list = []
      for frame_set, labels in zip(batch_samples['Filtered Frames'], batch_samples['Labels Encoded']):
        # Iterate through a row of the dataframe which has all the frames for one sample
        tensor_list = []
        for frame in frame_set:
          img = tf.io.read_file(str(frame)) # Load image via tf.io
          tensor = tf.io.decode_image(img, channels = 3, dtype = tf.dtypes.float32) # Convert to a tensor (specify 3 channels)
          tensor_list.append(tensor)
        # Stack the tensors to get shape (temporal length, height, width, channels)
        stacked_tensor_frames = tf.stack(tensor_list) 
        stacked_tensor_frames_list.append(stacked_tensor_frames)

        # Get the label information
        label_one_hot_encoded = tf.keras.utils.to_categorical(labels, self.num_classes)
        label_list.append(label_one_hot_encoded)

      # Convert the lists to numpy arrays to be fed into the neural network
      stacked_tensor_frames_list = tf.stack(stacked_tensor_frames_list)
      label_list = tf.stack(label_list)
      yield stacked_tensor_frames_list, label_list
