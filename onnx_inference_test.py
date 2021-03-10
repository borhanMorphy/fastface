import onnxruntime as rt
import imageio
import sys
import numpy as np

img = imageio.imread(sys.argv[1])[:,:,:3]
sess = rt.InferenceSession("test.onnx")

input_name = sess.get_inputs()[0].name

pred = sess.run(None, {input_name: [(np.transpose(img, (2,0,1))).astype(np.float32)]})

print(pred)