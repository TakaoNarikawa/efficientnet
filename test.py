import efficientnet.tfkeras as efn
import tensorflow as tf

inputs     = tf.ones([1, 450, 450, 3],  dtype=tf.dtypes.float32)
attention_inputs = tf.ones([1, 225, 225, 1], dtype=tf.dtypes.float32)

constructor = getattr(efn, f'EfficientNetB6')
model = constructor(include_top=False, weights='imagenet', input_shape=(450, 450, 3), attention_input_shape=(225, 225, 1), attention_operator="add")
x = model([inputs, attention_inputs])

model.summary()