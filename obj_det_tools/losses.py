import keras
import tensorflow as tf 

class SmoothL1Loss(tf.losses.Loss):
  """
    Implementation of smooth L1 loss (box loss).
  """
  def __init__(self, _delta):
    super().__init__(
        reduction="none",
        name="SmoothL1Loss"
    )
    self._delta = _delta

  def call(self, y_true, y_pred):
    abs_diff = tf.abs(y_true - y_pred)
    sq_diff = (abs_diff) ** 2
    loss = tf.where(
        tf.less(abs_diff, self._delta),
        0.5 * sq_diff,
        abs_diff - 0.5
    )
    return tf.reduce_sum(loss, axis=1)
  
class FocalLoss(tf.losses.Loss):
  """
    Implementation of focal loss (classification loss).
  """
  def __init__(self, _alpha, _gamma):
    super().__init__(
      reduction="none",
      name="FocalLoss"
    )
    self._alpha = _alpha
    self._gamma = _gamma

  def call(self, y_true, y_pred):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_true,
        logits=y_pred
    )
    prob = tf.nn.sigmoid(y_pred)
    alpha = tf.where(tf.equal(y_true, 1.0), 
                     self._alpha, 
                     (1.0 - self._alpha))
    pt = tf.where(tf.equal(y_true, 1.0),
                  prob,
                  1 - prob)
    loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy 
    return tf.reduce_sum(loss, axis=1)
