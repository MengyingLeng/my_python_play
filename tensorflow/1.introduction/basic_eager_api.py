import tensorflow as tf
import tensorflow.contrib.eager as tfe

import numpy as np

# eager可以立即执行图，不必建立完全后执行
print('star eager mode:')
tfe.enable_eager_execution()

print("Define constant tensors")
a = tf.constant(2)
print(a)




















