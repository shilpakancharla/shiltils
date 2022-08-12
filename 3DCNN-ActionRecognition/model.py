import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.layers import Conv3D, Dense, Dropout, Flatten, MaxPooling3D, GlobalAveragePooling3D

def define_3DCNN_model(num_labels = 10, input_shape = (None, None, None, None, 3)):
  model = Sequential()
  model.add(Conv3D(32, kernel_size = (3, 3, 3), input_shape = input_shape[1:], padding = 'same', activation = 'relu'))
  model.add(Conv3D(32, kernel_size = (3, 3, 3), padding = 'same', activation = 'relu'))
  model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'same'))
  model.add(Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', activation = 'relu'))
  model.add(Conv3D(64, kernel_size = (3, 3, 3), padding = 'same', activation = 'relu'))
  model.add(MaxPooling3D(pool_size = (3, 3, 3), padding = 'same'))

  model.add(GlobalAveragePooling3D())
  model.add(Dense(512, activation = 'relu'))
  model.add(Dropout(0.10))
  model.add(Dense(num_labels, activation = 'softmax'))
  model.compile(loss = categorical_crossentropy, 
                optimizer = Adam(lr = 0.0005), 
                metrics = ['accuracy'])
  model.summary()
  return model

def plot_history(history):
  """
    Plotting training and validation learning curves.

    Args:
      history: model history with all the metric measures
  """
  # Plot loss
  plt.title('Loss')
  plt.plot(history.history['loss'], color = 'blue', label = 'train')
  plt.plot(history.history['val_loss'], color = 'red', label = 'test')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation']) 
  plt.show()

  # Plot accuracy
  plt.title('Accuracy')
  plt.plot(history.history['accuracy'], color = 'blue', label = 'train')
  plt.plot(history.history['val_accuracy'], color = 'red', label = 'test')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'])
  plt.show()
