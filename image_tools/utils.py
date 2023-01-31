import cv2
import numpy as np

def rotate_image(image_path, angle):
  """
    Rotate image to ensure it is not cut off. 
  """
  image = cv2.imread(image_path) # Read the image
  (height, width) = image.shape[:2] 
  (cX, cY) = (width // 2, height // 2) # Get the center of the image

  # Get the rotation matrix (apply the negative of the angle to rotate 
  # clockwise), then get sine and cosine (rotation components of matrix)
  M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])

  # Compute new bounding dimensions of the image
  new_height = int((height * cos) + (width * sin))
  new_width = int((height * sin) + (width * cos))

  # Adjust rotation matrix to account translation
  M[0, 2] += (new_width / 2) - cX
  M[1, 2] += (new_height / 2) - cY

  # Perform rotation and return image
  return cv2.warpAffine(image, M, (new_width, new_height))
