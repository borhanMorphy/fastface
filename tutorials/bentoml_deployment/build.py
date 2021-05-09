# import fastface package to get pretrained model
import fastface as ff
import torch
import tempfile

# pretrained model
pretrained_model_name = "lffd_original"

# get pretrained model
model = ff.FaceDetector.from_pretrained(pretrained_model_name)

# export as onnx
opset_version = 11

dynamic_axes = {
    "input_data": {0: "batch", 2: "height", 3: "width"}, # write axis names
    "preds": {0: "batch"}
}

input_names = [
    "input_data"
]

output_names = [
    "preds"
]

# define dummy sample
input_sample = torch.rand(1, *model.arch.input_shape[1:])

# export model as onnx
with tempfile.NamedTemporaryFile(suffix='.onnx', delete=True) as tmpfile:
    model.to_onnx(tmpfile.name,
        input_sample=input_sample,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_params=True
    )

    # get FaceDetectionService
    from service import FaceDetectionService

    # create FaceDetectionService instance
    face_detection_service = FaceDetectionService()

    # Pack the model artifact
    face_detection_service.pack('model', tmpfile.name)

# Save the service to disk for model serving
saved_path = face_detection_service.save(version="v{}".format(ff.__version__))

print("saved path: {}".format(saved_path))