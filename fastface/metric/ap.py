from pytorch_lightning.metrics import Metric
from typing import List
import torch
from .functional import average_precision

class AveragePrecision(Metric):
	"""pytorch_lightning.metrics.Metric instance to calculate binary average precision
	Args:
		iou_threshold (float): AP score IoU threshold, default is 0.5
	"""

	def __init__(self, iou_threshold: float = 0.5, threshold_steps: int = 1000):
		super().__init__(dist_sync_on_step=False, compute_on_step=False)

		self.iou_threshold = iou_threshold
		self.threshold_steps = threshold_steps
		# [Ni,5 dimensional as xmin,ymin,xmax,ymax,conf]
		self.add_state("pred_boxes", default=[], dist_reduce_fx=None)
		# [Ni,4 dimensional as xmin,ymin,xmax,ymax]
		self.add_state("target_boxes", default=[], dist_reduce_fx=None)

	# pylint: disable=method-hidden
	def update(self, preds: List[torch.Tensor], targets: List[torch.Tensor]):
		"""
		Arguments:
			preds (List) -- [Ni,5 dimensional as xmin,ymin,xmax,ymax,conf]
			targets (List) -- [Mi,4 dimensional as xmin,ymin,xmax,ymax]
		"""
		# pylint: disable=no-member
		if isinstance(preds, List): self.pred_boxes += preds
		else: self.pred_boxes.append(preds)

		if isinstance(targets, List): self.target_boxes += targets
		else: self.target_boxes.append(targets)

	# pylint: disable=method-hidden
	def compute(self):
		"""Calculates average precision
		"""
		# N,3 as iou,best,confidence with sorted by confidence
		# pylint: disable=no-member

		return average_precision(self.pred_boxes, self.target_boxes)
