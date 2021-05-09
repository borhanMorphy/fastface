from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import ImageInput
from bentoml.frameworks.onnx import OnnxModelArtifact

import numpy as np
from typing import List, Dict

@env(infer_pip_packages=True)
@artifacts([
    OnnxModelArtifact('model', backend="onnxruntime")
])
class FaceDetectionService(BentoService):

    def prepare_input(self, img: np.ndarray) -> np.ndarray:
        img = np.transpose(img[:, :, :3], (2, 0, 1))
        return np.expand_dims(img, axis=0).astype(np.float32)

    def to_json(self, results: np.ndarray) -> Dict:
        # results: (N, 6) as x1,y1,x2,y2,score,batch_idx
        return {
            "boxes": results[:, :4].astype(np.int32).tolist(),
            "scores": results[:, 4].astype(np.float32).tolist()
        }

    @api(input=ImageInput(), batch=True, mb_max_batch_size=8, mb_max_latency=1000)
    def detect(self, imgs: List[np.ndarray]):
        input_name = self.artifacts.model.get_inputs()[0].name
        preds = []
        for img in imgs:
            results = self.artifacts.model.run(None, {input_name: self.prepare_input(img) })[0]
            preds.append(
                self.to_json(results)
            )
        return preds