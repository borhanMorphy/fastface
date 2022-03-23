import argparse

from cv2 import cv2
import numpy as np
import torch

import fastface as ff


def get_arguments():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        "-m",
        type=str,
        default="lffd_original",
        help=f"pretrained models {','.join(ff.list_pretrained_models())} or checkpoint path",
    )

    ap.add_argument(
        "--device",
        "-d",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    ap.add_argument("--input", "-i", type=str, required=True, help="image file path")

    ap.add_argument(
        "--det-threshold",
        "-dt",
        type=float,
        default=0.7,
        help="detection score threshold",
    )

    ap.add_argument(
        "--iou-threshold", "-it", type=float, default=0.4, help="iou score threshold"
    )

    ap.add_argument(
        "--target-size",
        "-t",
        type=int,
        default=None,
        help="interpolates all inputs to given target size",
    )

    return ap.parse_args()


def load_image(img_path: str) -> np.ndarray:
    """loads rgb image using given file path

    Args:
        img_path (str): image file path to load

    Returns:
        np.ndarray: rgb image as np.ndarray
    """
    img = cv2.imread(img_path)

    if img.shape[2] == 4:
        # found RGBA
        img = img[:, :, :3]

    return img[..., [2, 1, 0]]


def main(
    model: str,
    device: str,
    img_path: str,
    det_threshold: float,
    iou_threshold: float,
    target_size: int,
):

    # load image
    img = load_image(img_path)

    # get pretrained model
    model = ff.FaceDetector.from_pretrained(model)

    # get model summary
    model.summarize()

    # set model to eval mode
    model.eval()

    # move model to given device
    model.to(device)

    # model feed forward
    (preds,) = model.predict(
        img,
        det_threshold=det_threshold,
        iou_threshold=iou_threshold,
        target_size=target_size,
    )

    # visualize predictions
    pretty_img = ff.utils.vis.render_predictions(img, preds)

    # show image
    pretty_img.show()


if __name__ == "__main__":
    # python demo.py -m lffd_original -d cuda -t 640 -i <img_file_path>
    args = get_arguments()
    main(
        args.model,
        args.device,
        args.input,
        args.det_threshold,
        args.iou_threshold,
        args.target_size,
    )
