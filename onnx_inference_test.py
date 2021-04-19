import onnxruntime as ort
import imageio
import sys
from PIL import Image, ImageDraw, ImageColor
import random
import numpy as np
from typing import Tuple, List, Dict

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

img = imageio.imread(sys.argv[1])[:,:,:3]
sess = ort.InferenceSession("lffd_original.onnx")

input_name = sess.get_inputs()[0].name

o_batch_ids, o_preds = sess.run(None, {input_name: [(np.transpose(img, (2,0,1))).astype(np.float32)]})

boxes = o_preds[:, :4].astype(np.int32).tolist()
scores = o_preds[:, 4].tolist()

# visualize predictions
pretty_img = prettify_detections(img, {
    'boxes': boxes,
    'scores': scores
})

# show image
pretty_img.show()