from typing import Tuple, Dict
import onnxruntime as ort
import imageio
import sys
from PIL import Image, ImageDraw, ImageColor
import random
import numpy as np
import tempfile
import fastface as ff
import torch

def prettify_detections(img: np.ndarray, preds: Dict,
        color: Tuple[int, int, int] = None) -> Image:
    """
    Args:
        img (np.ndarray): 3 channeled image
        preds (Dict): predictions as {'boxes':[[x1,y1,x2,y2], ...], 'scores':[<float>, ..]}
        color (Tuple[int,int,int], optional): color of the boundaries. if None that it will be random color.

    Returns:
        Image: 3 channeled pil image
    """
    color = random.choice(list(ImageColor.colormap.keys()))
    pil_img = Image.fromarray(img)
    for (x1,y1,x2,y2), score in zip(preds['boxes'],preds['scores']):
        ImageDraw.Draw(pil_img).rectangle([(x1,y1),(x2,y2)], outline=color, width=3)
    return pil_img


def get_ort_sess():
    module = ff.FaceDetector.from_pretrained("lffd_slim").eval()
    opset_version = 12
    dynamic_axes = {
        "input_data": {0: "batch", 2: "height", 3: "width"}, # write axis names
        "preds": {0: "batch"}
    }
    input_names = [
        "input_data"]

    output_names = [
        "preds"
    ]

    input_sample = torch.rand(1,*module.arch.input_shape[1:])

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=True) as tmpfile:

        module.to_onnx(tmpfile.name,
            input_sample=input_sample,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_params=True
        )
        sess = ort.InferenceSession(tmpfile.name)
    return sess

sess = get_ort_sess()
img = imageio.imread(sys.argv[1])[:,:,:3]

input_name = sess.get_inputs()[0].name
o_preds = sess.run(None, {input_name: [(np.transpose(img, (2,0,1))).astype(np.float32)]})
print(o_preds[0].shape)
exit(0)
o_preds, = sess.run(None, {input_name: [(np.transpose(img, (2,0,1))).astype(np.float32)]})

boxes = o_preds[:, :4].astype(np.int32).tolist()
scores = o_preds[:, 4].tolist()

# visualize predictions
pretty_img = prettify_detections(img, {
    'boxes': boxes,
    'scores': scores
})

# show image
pretty_img.show()