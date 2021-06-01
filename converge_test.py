import fastface as ff
from cv2 import cv2
import numpy as np
import torch

loss_fn = ff.loss.DIoULoss()
canvas_size = 300

canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
gt = torch.rand(1, 4) * canvas_size

order = gt.argsort(descending=False)
gt = gt[:, order]

x1,y1,x2,y2 = gt.flatten().long().numpy()
cv2.rectangle(canvas, (x1,y1) ,(y2,y2), (0,255,0))
cv2.imshow("", canvas)
cv2.waitKey(0)