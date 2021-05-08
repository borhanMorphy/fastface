Export
======


To Onnx
++++++++++++++

.. code-block:: python

    import fastface as ff
    import torch

    # checkout available pretrained models
    print(ff.list_pretrained_models())
    # ["lffd_slim", "lffd_original"]

    pretrained_model_name = "lffd_slim"

    # build pl.LightningModule using pretrained weights
    model = ff.FaceDetector.from_pretrained(pretrained_model_name)

    # onnx export configs
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
    model.to_onnx("{}.onnx".format(pretrained_model_name),
        input_sample=input_sample,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        export_params=True
    )

To TorchScript
+++++++++++++++++++++

.. code-block:: python

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