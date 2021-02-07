# FastFace BentoML Deployment
**[BentoML](https://www.bentoml.ai/) is a model serving framework, enabling to deliver prediction services in a fast, repeatable, and scalable way.<br>
This tutorial will explain how to use bentoml to deploy [fastface](https://github.com/borhanMorphy/light-face-detection) models into production as a service.**

## Installation
**install latest fastface and bentoml via pip**
```
pip install fastface BentoML==0.11.0 -U
```

## BentoService Definition
define BentoService as [service.py](./service.py)  
```python
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import ImageInput
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.json_file import JSONArtifact
from bentoml.service.artifacts.common import PickleArtifact

import numpy as np
from typing import List
import torch

@env(pip_dependencies=["fastface"])
@artifacts([
    PytorchModelArtifact('model'),
    JSONArtifact('config'),
    PickleArtifact('preprocess')
])
class FaceDetectionService(BentoService):

    @api(input=ImageInput(), batch=True, mb_max_batch_size=8, mb_max_latency=1000)
    def detect(self, imgs:List[np.ndarray]):
        # initialize model if not initialized yet
        if not self.artifacts.config['initialized']:
            # set model to eval mode
            self.artifacts.model.eval()
            # move model to specifed device
            self.artifacts.model.to(self.artifacts.config['device'])

            # enable tracking to perform postprocess after inference 
            self.artifacts.preprocess.enable_tracking()

            # set initialize flag true
            self.artifacts.config['initialized'] = True

        # apply transforms
        imgs = torch.stack([
            self.artifacts.preprocess(image) for image in imgs], dim=0
            ).to(self.artifacts.config['device'])

        preds:List = []

        for pred in self.artifacts.model.predict(imgs):
            # postprocess to adjust predictions
            pred = self.artifacts.preprocess.adjust(pred.cpu().numpy())
            # pred np.ndarray(N,5) as x1,y1,x2,y2,score
            payload = [{'box':person[:4].astype(np.int32).tolist(), 'score':person[4]} for person in pred]
            preds.append(payload)

        # reset queue
        self.artifacts.preprocess.flush()

        return preds
```

## Build And Pack BentoService
define operations as [build.py](./build.py)
```python
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
```

run `build.py` with the following
```
python build.py
```

## Serving The Model In Production Mode

To serve model in production mode run the following (model will be served from http://0.0.0.0:5000).
```
bentoml serve-gunicorn FaceDetectionService:latest -w 1
```

## Test Rest API

test rest api with [test.py](./test.py)
```python
import requests
from fastface.utils.visualize import prettify_detections
import imageio

url = "http://localhost:5000/detect"

payload={}
files=[
  ('image',('friends2.jpg',open('../../resources/friends2.jpg','rb'),'image/jpeg'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.json())

pretty_img = prettify_detections(imageio.imread('../../resources/friends2.jpg'), response.json())

# show image
pretty_img.show()
```

Output should look like this<br>


![alt text](../../resources/friends2.jpg)

## Build And Deploy Using Docker
BentoML also provides docker support for distributing services.<br>

Run following to build docker image
```
docker build --tag face-detection-service $HOME/bentoml/repository/FaceDetectionService/v0.1.0rc1/
```

After docker image build is done, run docker container with the following
```
docker run -p 5000:5000 -e BENTOML__APISERVER__DEFAULT_GUNICORN_WORKER_COUNTS=1 --name face_detection_service face-detection-service
```