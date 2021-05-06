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