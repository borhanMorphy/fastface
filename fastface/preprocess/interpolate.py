import torch
from typing import Tuple

__all__ = ['Interpolate']

class Interpolate(torch.nn.Module):
	max_size: int
	mode: str
	align_corners: bool

	def __init__(self, max_size: int = 640, mode: str = "bilinear", align_corners: bool = False):
		super().__init__()
		self.mode = mode
		self.align_corners = align_corners
		self.max_size = max_size

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Upsamples or Downsamples given image
		with shape of (batch_size, channel, height, width)

		Arguments:
			x (torch.Tensor): (batch_size,channel,height,width)

		Returns:
			Tuple[torch.Tensor, torch.Tensor]:
				(0): ouput image as tensor with shape of (batch_size,channel,scaled_height,scaled_width)
				(1): scale factor as (1,) dimensional tensor

		>>> import torch
		>>> import fastface as ff
		>>> interpolate = ff.preprocess.Interpolate(max_size=100)
		>>> img = torch.rand(1,3,50,22)
		>>> scaled_img, scale_factor = interpolate(img)
		>>> scaled_img.shape
		torch.Size([1, 3, 100, 44])
		>>> scale_factor.shape
		torch.Size([1])
		>>> scaled_img.shape[2] == (scale_factor * img.shape[2]).floor()
		tensor([True])
		>>> scaled_img.shape[3] == (scale_factor * img.shape[3]).floor()
		tensor([True])
		"""

		image_w = x.size(3)
		image_h = x.size(2)

		scale_factor: float = self.max_size / max(image_h, image_w)
		x = torch.nn.functional.interpolate(x,
			scale_factor=scale_factor, mode=self.mode,
			align_corners=self.align_corners, recompute_scale_factor=True)

		return x, torch.tensor([scale_factor], device=x.device, dtype=x.dtype) # pylint: disable=not-callable

class DummyInterpolate(torch.nn.Module):

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Mimics Interpolate but returns same image with scale factor 1
		with shape of (batch_size, channel, height, width)

		Arguments:
			x (torch.Tensor): (batch_size,channel,height,width)

		Returns:
			Tuple[torch.Tensor, torch.Tensor]:
				(0): ouput image as tensor with shape of (batch_size, channel, height, width)
				(1): scale factor as (1,) dimensional tensor

		>>> import torch
		>>> import fastface as ff
		>>> interpolate = ff.preprocess.DummyInterpolate()
		>>> img = torch.rand(1,3,75,65)
		>>> scaled_img, scale_factor = interpolate(img)
		>>> scaled_img.shape
		torch.Size([1, 3, 75, 65])
		>>> scale_factor.shape
		torch.Size([1])
		"""
		return x, torch.tensor([1], device=x.device, dtype=x.dtype) # pylint: disable=not-callable
