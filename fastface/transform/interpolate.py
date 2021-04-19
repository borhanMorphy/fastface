import numpy as np
from typing import List,Tuple,Union,Any
from .transform import Transform
from PIL import Image

class Interpolate(Transform):
    """Interpolates the image and boxes using maximum dimension
    """

    def __init__(self, max_dim:int=640):
        super(Interpolate,self).__init__()
        self.max_dim = max_dim

    def __call__(self, img:np.ndarray,
            gt_boxes:np.ndarray=None) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        h,w = img.shape[:2]

        sf = self.max_dim / max(h,w)

        nh = int(sf*h)
        nw = int(sf*w)

        if self.tracking: self.register_op({'scale_factor':sf})

        nimg = np.array(Image.fromarray(img).resize((nw,nh)))

        if isinstance(gt_boxes, type(None)): return nimg

        ngt_boxes_boxes = gt_boxes * sf
        return nimg,ngt_boxes_boxes

    def adjust(self, pred_boxes:np.ndarray, scale_factor:float=1.) -> np.ndarray:
        pred_boxes[:, :4] = pred_boxes[:, :4] / scale_factor
        return pred_boxes