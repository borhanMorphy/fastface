from cv2 import cv2
import onnxruntime as ort
import time

cap = cv2.VideoCapture(0)
sess = ort.InferenceSession("lffd_slim.onnx")
import numpy as np

input_name = sess.get_inputs()[0].name

while 1:
    ret, frame = cap.read()
    start_time = time.time()

    if not ret:
        break

    batch = np.expand_dims((np.transpose(frame, (2, 0, 1))).astype(np.float32), axis=0)
    #batch = np.concatenate([batch, batch.copy()], axis=0)

    preds, = sess.run(None, {input_name: batch})

    boxes = preds[:, :4].astype(np.int32).tolist()
    scores = preds[:, 4].tolist()

    for x1,y1,x2,y2 in boxes:
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imshow("", frame)
    print("FPS: {}".format(1/(time.time() - start_time)))
    if cv2.waitKey(1) == 27:
        break