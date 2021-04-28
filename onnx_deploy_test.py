import fastface as ff
import torch

model_name = "lffd_slim"
model = ff.FaceDetector.from_pretrained(model_name).eval()
input_shape = model.hparams["config"]["input_shape"][1:]
file_path = "{}.onnx".format(model_name)
input_sample = torch.rand(1, *input_shape)
opset_version = 12
dynamic_axes = {
    "input_data": {0: "batch", 2: "height", 3: "width"}, # write axis names
    "preds": {0: "batch"}
}
input_names = [
    "input_data"]

output_names = [
    "preds"
]

verbose = True

model.to_onnx(
    file_path=file_path,
    input_sample=input_sample,
    opset_version=opset_version,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    verbose=verbose
)
