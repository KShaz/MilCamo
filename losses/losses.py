import tensorflow as tf

class NewMyLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth
        self.bceloss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def dice_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score  # Dice Loss = 1 - Dice Coefficient

    def iou_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        total = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        union = total - intersection

        iou_score = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou_score  # IoU Loss = 1 - IoU Score

    def call(self, y_true, y_pred):
        bce = self.bceloss(y_true, y_pred)
        dice = self.dice_loss(y_true, y_pred)
        iou = self.iou_loss(y_true, y_pred)

        return bce + dice + iou
