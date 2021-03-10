import fastface as ff
import torch

model = ff.FaceDetector.from_pretrained("lffd_original")
input_shape = model.hparams["config"]["input_shape"][1:]
file_path="test.onnx"
input_sample=torch.rand(1,*input_shape)
opset_version=12
dynamic_axes = {
    "input_data": {0: "batch", 2: "height", 3: "width"}, # write axis names
    "batch_ids": {0: "batch"},
    "preds": {0: "batch"}
}
input_names = [
    "input_data",
    "iou_threshold",
    "det_threshold"
]
output_names = [
    "batch_ids",
    "preds"
]
verbose = False

torch.onnx.export(model, input_sample, file_path,
    opset_version=opset_version, input_names=input_names,
    output_names=output_names, dynamic_axes=dynamic_axes, verbose=verbose)

"""
model.to_onnx(
    file_path=file_path,
    input_sample=input_sample,
    opset_version=opset_version,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    verbose=verbose
)
"""