import efficientnet.tfkeras as efn
import tensorflow as tf

inputs     = tf.ones([1, 450, 450, 3],  dtype=tf.dtypes.float32)
attention_inputs = tf.ones([1, 450, 450, 1], dtype=tf.dtypes.float32)
attention_inputs2 = tf.ones([1, 450, 450, 1], dtype=tf.dtypes.float32)

constructor = getattr(efn, f'EfficientNetB6')
model = constructor(include_top=False, weights='imagenet', input_shape=(450, 450, 3), attention_input_shape=(450, 450, 1), attention_input_double=True)
x = model([inputs, attention_inputs, attention_inputs2])

model.summary()