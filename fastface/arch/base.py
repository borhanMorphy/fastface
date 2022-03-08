import abc
from typing import Any, Dict, Tuple

import torch

class ArchInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            # required functions
            hasattr(subclass, "forward")
            and callable(subclass.forward)
            and hasattr(subclass, "compute_preds")
            and callable(subclass.compute_preds)
            and hasattr(subclass, "compute_loss")
            and callable(subclass.compute_loss)
            and hasattr(subclass, "build_targets")
            and callable(subclass.build_targets)
            and hasattr(subclass, "configure_optimizers")
            and callable(subclass.configure_optimizers)
            and hasattr(subclass, "input_shape")
            or NotImplemented
        )

    @abc.abstractmethod
    def forward(self, batch: torch.Tensor) -> Any:
        """Compute logits with given normalized batch of images

        Args:
            batch (torch.Tensor): normalized bach of images with shape of B x C x H x W

        Returns:
            Any: logits with architecture spesific shape or type
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_preds(self, logits: Any) -> torch.Tensor:
        """Compute predictions with given logits

        Args:
            logits (Any): logits with architecture spesific shape or type (`forward` output)

        Returns:
            torch.Tensor: model predictions as tensor with shape of B x N x (5 + 2*l)
                xmin, ymin, xmax, ymax, score, *landmarks
                where `l` is number of landmarks.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, logits: Any, target_logits: Any, hparams: Dict = None) -> Dict[str, torch.Tensor]:
        """Compute losses with given logits

        Args:
            logits (Any): logits with architecture spesific shape or type (`forward` output)
            target_logits (Any): target logits with architecture spesific shape or type (`build_targets` output)
            hparams (Dict, optional): model hyperparameter dict. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: loss values with key names. Must always contain `loss` key
        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_targets(self, batch: torch.Tensor, raw_targets: Any) -> Any:
        """Compute target logits with given inputs and targets. Targets depends on used `collate_fn` function

        Args:
            batch (torch.Tensor): normalized bach of images with shape of B x C x H x W
            raw_targets (Any): collected targets from batch, type depends used `collate_fn` function

        Returns:
            Any: target logits with architecture spesific shape or type (same as logits)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def configure_optimizers(self, hparams: Dict = None):
        """Returns optimizers using given optimizer configurations

        Args:
            hparams (Dict, optional): Optimizer configurations captured from hyperparameters. Defaults to None.

        Returns:
            Any: Defined optimizers for the model
        """
        raise NotImplementedError

    @abc.abstractproperty
    def input_shape(self) -> Tuple[int, int, int]:
        """Preferred input shape for the model

        Returns:
            Tuple[int, int, int]: input shape with order of (channel, height, width)
        """
        raise NotImplementedError
