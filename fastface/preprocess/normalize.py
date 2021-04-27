import torch
from typing import List, Union

__all__ = ['Normalize']

class Normalize(torch.nn.Module):
	mean: torch.Tensor
	std: torch.Tensor
	channels: int

	def __init__(self,
			mean: Union[List, float, int] = 0,
			std: Union[List, float, int] = 1,
			channels: int = 3):

		super(Normalize, self).__init__()
		if isinstance(mean, list):
			assert len(mean) == channels, f"mean dimension must be {channels} not {len(mean)}"
			mean = [float(m) for m in mean]
		else:
			mean = [float(mean) for _ in range(channels)]

		if isinstance(std, list):
			assert len(std) == channels, f"std dimension must be {channels} not {len(std)}"
			std = [float(m) for m in std]
		else:
			std = [float(std) for _ in range(channels)]

		self.register_buffer(
			"mean",
			torch.tensor(mean).view(-1, 1, 1), # pylint: disable=not-callable
			persistent=False
		)

		self.register_buffer(
			"std",
			torch.tensor(std).view(-1, 1, 1), # pylint: disable=not-callable
			persistent=False
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Applies `(x - mean) / std`

		Arguments:
			x (torch.Tensor) -- tensor with shape of (batch_size,channel,height,width)

		Returns:
			torch.Tensor -- tensor with shape of (batch_size,channel,height,width)

		>>> import torch
		>>> import objectdetection as od
		>>> normalize = od.preprocess.Normalize(mean=0, std=1)
		>>> rgb_image = torch.rand(1,3,20,20)
		>>> normalized_rgb_image = normalize(rgb_image)
		>>> normalized_rgb_image.shape
		torch.Size([1, 3, 20, 20])
		"""
		normalized_x = x - self.mean
		normalized_x = normalized_x / self.std
		return normalized_x
