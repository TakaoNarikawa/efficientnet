import efficientnet.tfkeras as efn
import tensorflow as tf

inputs     = tf.ones([1, 450, 450, 3],  dtype=tf.dtypes.float32)
seg_inputs = tf.ones([1, 225, 225, 56], dtype=tf.dtypes.float32)

constructor = getattr(efn, f'EfficientNetB6')
model = constructor(include_top=False, weights='imagenet', input_shape=(450, 450, 3), seg_input_shape=(225, 225, 56))
x = model([inputs, seg_inputs])

model.summary()