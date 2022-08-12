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

def split_video_to_frames(src_video_path, dest_img_path, frame_subsample_factor):
  """
  Split video into frames.

  Args:
    src_video_path: Video file path to get frames from.
    dest_img_path: Path to place images in.
    frame_subsample_factor: Sampling every number of frames such that we do not use the entire video.
  """
  # Check if the destination folder already exists
  if not pathlib.Path(dest_img_path).exists():
    pathlib.Path(dest_img_path).mkdir(parents = True, exist_ok = True)
  
  # Read each frame by frame
  src = cv2.VideoCapture(str(src_video_path))
  count = 0
  while True:
    ret, frame = src.read()
    if frame is None:
      break
    if count % frame_subsample_factor == 0:
      cv2.imwrite(str(dest_img_path/f"frame{count:04d}.jpg"), frame)
    count += 1
  src.release()
  cv2.destroyAllWindows()

def train_val_test_split(df):
  """
    Split the dataframe into training, validation, and test sets randomly.
    
    Args:
      df: Dataframe with features and labels
      
    Returns:
      train: Training set dataframe with reference to features and label
      val: Validation set dataframe with reference to features and label
      test: Test set dataframe with reference to features and label
  """
  train, val, test = np.split(df.sample(frac = 1, random_state = 1), [int(.6 * len(df)), int(.8 * len(df))])
  return train, val, test

def create_frames_labels_mapping(data_directory):
  """
    Create the mapping of the frames to their respective folders that represent one video and their label.

    Args:
      data_directory: Where frames are located in the file path.
      
    Return:
      frames_label_df: Dataframe of frames associated with their respective labels.
  """
  root_directory = pathlib.Path(data_directory)
  frames_for_video = collections.defaultdict(list)
  for frame_path in root_directory.glob('*/*/*.jpg'):
    frames_for_video[frame_path.parent].append(frame_path)
 
  frames_label_df = pd.DataFrame(
      ([sorted(frames_for_video[x]), x.name.split('_')[1]] for x in frames_for_video),
      columns = ['Frame List', 'Label']
  )

  return frames_label_df

def get_random_subsection(frame_list, temporal_length):
  """
    Returns a random subsection of frames.

    Args:
      frame_list: List of frames for a particular video (can be Pandas series data type passed in).
      temporal_length: Length of the frames desired.

    Return:
      Subsection of the frames from a random starting point of the specified temporal length.
  """
  if temporal_length == len(frame_list):
    return frame_list
  
  last_idx = len(frame_list) - 1
  idx = len(frame_list) // 2
  # Generate two random numbers for start and halfway point of list that must have a difference of temporal length
  start = random.randint(0, idx)
  if start + temporal_length > last_idx:
    # Backtrack such that you get the section within the length of the list
    start = (start + temporal_length) - last_idx

  return frame_list[start:start + temporal_length]

def filter_frames(frames_label_df, temporal_length = 50):
  """
    Returns a dataframe with column of frames that are all of the same temporal length. The newly returned
    set of filtered frames are a random subsection of the original list of frames for a video.

    Args:
      frames_label_df: Dataframe with list of frames (differing lengths between rows) and their associated label.
      temporal_length: Number of frames from each video.

    Return:
      frames_label_df: Dataframe that includes the original contents of the parameter passed in and a column of lists of frames of the temporal length.
  """  
  # If the list of frames is longer than the temporal length, delete those excess frames
  frames_label_df['Filtered Frames'] = None
  for index, inst in frames_label_df.iterrows():
    filtered_inst = get_random_subsection(inst['Frame List'], temporal_length)
    if len(filtered_inst) != temporal_length:
      filtered_inst = inst['Frame List'][:temporal_length]
    frames_label_df.at[index, 'Filtered Frames'] = filtered_inst

  return frames_label_df

def encode_labels_of_frames(frames_label_df):
  """
    Adds a column of encoding for the categorical values of the category that the frames belong to.

    Args:
      frames_label_df: Dataframe with list of frames and their associated label.

    Return:
      frames_label_df: Dataframe that includes a column of encoded label.
  """
  frames_label_df['Label'] = frames_label_df['Label'].astype('category')
  frames_label_df['Labels Encoded'] = frames_label_df['Label'].cat.codes
  return frames_label_df
