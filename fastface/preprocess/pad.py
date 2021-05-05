import torch
from typing import Tuple

__all__ = ['Pad']

class Pad(torch.nn.Module):
	target_size: Tuple[int, int]
	mode: str
	pad_value: float

	def __init__(self, target_size: Tuple, mode: str = "constant", pad_value: float = .0):
		super(Pad, self).__init__()
		self.target_size = target_size # [0] height , [1] width
		self.mode = mode
		self.pad_value = pad_value

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Applies padding to given data using `pad_value` and `target_size`

		Arguments:
			x (torch.Tensor): tensor with shape of (batch_size,channel,height,width)

		Returns:
			Tuple[torch.Tensor, torch.Tensor]:
				(0): ouput data as tensor with shape of (batch_size,channel,target_size[0],target_size[1])
				(1): padding value as (4,) dimensional tensor with order of
					(pad_left, pad_top, pad_right, pad_bottom)

		>>> import torch
		>>> import fastface as ff
		>>> pad = ff.preprocess.Pad((88, 77))
		>>> data = torch.rand(1,1,80,70)
		>>> padded_data, paddings = pad(data)
		>>> padded_data.shape
		torch.Size([1, 1, 88, 77])
		>>> paddings.shape
		torch.Size([4])
		>>> padded_data.shape[2] == (data.shape[2] + paddings[1] + paddings[3])
		tensor(True)
		>>> padded_data.shape[3] == (data.shape[3] + paddings[0] + paddings[2])
		tensor(True)
		"""
		dh, dw = self.target_size
		w = x.size(3)
		h = x.size(2)
		sub_w = int((dw - w)//2)
		sub_h = int((dh - h)//2)
		pad_left, pad_right = sub_w, sub_w

		pad_left += (dw - w) % 2

		pad_top, pad_bottom = sub_h, sub_h

		pad_top += (dh - h) % 2

		if min(pad_left, pad_right, pad_top, pad_bottom) < 0:
			return x, torch.tensor([0, 0, 0, 0], device=x.device, dtype=x.dtype)	# pylint: disable=not-callable  # noqa

		x = torch.nn.functional.pad(x, [pad_left, pad_right, pad_top, pad_bottom],
			mode=self.mode, value=self.pad_value)

		return x, torch.tensor([pad_left, pad_top, pad_right, pad_bottom], device=x.device, dtype=x.dtype)	# pylint: disable=not-callable  # noqa

class DummyPad(torch.nn.Module):

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Mimics Pad but returns same image with paddings [0,0,0,0]
		with shape of (batch_size, channel, height, width)

		Arguments:
			x (torch.Tensor): tensor with shape of (batch_size,channel,height,width)

		Returns:
			Tuple[torch.Tensor, torch.Tensor]:
				(0): ouput data as tensor with shape of (batch_size,channel,target_size[0],target_size[1])
				(1): padding value as (4,) dimensional tensor with order of
					(pad_left, pad_top, pad_right, pad_bottom)

		>>> import torch
		>>> import fastface as ff
		>>> pad = ff.preprocess.Pad((88, 77))
		>>> data = torch.rand(1,1,80,70)
		>>> padded_data, paddings = pad(data)
		>>> padded_data.shape
		torch.Size([1, 1, 88, 77])
		>>> paddings.shape
		torch.Size([4])
		"""
		return x, torch.tensor([0, 0, 0, 0], device=x.device, dtype=x.dtype) # pylint: disable=not-callable
