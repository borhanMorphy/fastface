import fastface as ff
import torch

# checkout available pretrained models
print(ff.list_pretrained_models())
# ["lffd_slim", "lffd_original"]

pretrained_model_name = "lffd_slim"

# build pl.LightningModule using pretrained weights
model = ff.FaceDetector.from_pretrained(pretrained_model_name)

model.eval()

sc_model = model.to_torchscript()

torch.jit.save(sc_model, "{}.ts".format(pretrained_model_name))
