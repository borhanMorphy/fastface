import fastface as ff
import torch

model = ff.FaceDetector.from_pretrained("lffd_original")

sc_model = model.to_onnx(
    file_path="test.onnx",
    input_sample=[torch.rand(3,500,480)],
    opset_version=12,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "sequence"}, "output": {0: "sequence"}}
)