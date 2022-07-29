import os
import cv2
import rarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

def extract_content(rar_obj, output_path) -> None:
  """
    If the filename already exists in the working directory, skip over extracting it.
    If it does not exist, then extract it.
    
    Args:
      rar_obj: rar file object to unzip.
      output_path: Path for compressed data to be unzipped.
  """
  for f in rar_obj.namelist(): # Check if file exists in output path
  if not os.path.exists(path = output_path + f): # If it does not exist, extract it
    rar_obj.extract(f)
    print("Extracted " + f)
    
def convert_to_greyscale(src_video_path, dest_video_path) -> None:
  """
    Convert a color video to a greyscale video.
    
    Args:
      src_video_path: Original color video file path.
      dest_video_path: Path where greyscale video is created. 
  """
  src = cv2.VideoCapture(src_video_path) # Read in the color video
  # Set the resolutions
  frame_width = int(src.get(3))
  frame_height = int(src.get(4))
  size = (frame_width, frame_height)
  dest = cv2.VideoWriter(dest_video_path,
                         cv2.VideoWriter_fourcc(*'DIVX'),
                         10, 
                         size, 
                         0)
  
  while True:
    ret, frame = src.read()
    if frame is None:
      break
    else:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      dest.write(gray) # Save the video

  src.release()
  cv2.destroyAllWindows()
  
def split_video_to_frames(src_video_path, dest_img_path, fps) -> None:
  """
    Split video into frames.

    Args: 
      src_video_path: Video file path to get frames from
      dest_img_path: Path to place images in
      fps: Frames per second to create images
  """
  # Get the directory name from the src_video_path
  if os.path.exists(dest_img_path):
    print(dest_img_path + " already exists. Do not create frames.")
  else:
    # Create the directory
    print("Creating " + dest_img_path)
    os.makedirs(dest_img_path)
    src = cv2.VideoCapture(src_video_path)
    count = 0
    while True:
      ret, frame = src.read()
      if frame is None:
        break
      else:
        cv2.imwrite(dest_img_path + "frame%d.jpg" % count, frame)
        print("Created " + dest_img_path + "frame%d.jpg" % count)
        count += 1

    src.release()
    cv2.destroyAllWindows()

def train_val_test_split(df):
  """
    Split the dataframe into training, validation, and test sets randomly.
    
    Args:
      df: Dataframe with features and labels
      
    Returns:
      train: training set dataframe with reference to features and label
      val: validation set dataframe with reference to features and label
      test: test set dataframe with reference to features and label
  """
  train, val, test = np.split(df.sample(frac = 1, random_state = 1), [int(.6 * len(df)), int(.8 * len(df))])
  return train, val, test
