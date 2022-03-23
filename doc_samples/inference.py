from cv2 import cv2

import fastface as ff

# checkout available pretrained models
print(ff.list_pretrained_models())
# ["lffd_slim", "lffd_original"]

# build pl.LightningModule using pretrained weights
model = ff.FaceDetector.from_pretrained("lffd_slim")

# set model to eval mode
model.eval()

# load image
img = cv2.imread("<your_img_file_path>")[..., [2, 1, 0]]

# find faces
(preds,) = model.predict(img)
"""preds
{
    'boxes': [[xmin, ymin, xmax, ymax], ...],
    'scores':[<float>, ...]
}
"""

# visualize predictions
pil_img = ff.utils.vis.render_predictions(img, preds)
pil_img.show()
