# import fastface package to get pretrained model
import fastface as ff
import torch

# get pretrained model
model = ff.module.from_pretrained("original_lffd_560_25L_8S")
# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get FaceDetectionService
from service import FaceDetectionService

# create FaceDetectionService instance
face_detection_service = FaceDetectionService()

# Pack the newly trained model artifact
face_detection_service.pack('model', model.arch)
face_detection_service.pack('preprocess', model.preprocess)
face_detection_service.pack('config', {
    'device': device,
    'initialized': False
})

# Save the service to disk for model serving
saved_path = face_detection_service.save(version="v{}".format(ff.__version__))

print("saved path: {}".format(saved_path))