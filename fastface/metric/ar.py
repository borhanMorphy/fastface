from pytorch_lightning.metrics import Metric
from typing import List
import torch
from ..utils.box import jaccard_vectorized
from scipy.optimize import linear_sum_assignment

class AverageRecall(Metric):
	r"""pytorch_lightning.metrics.Metric instance to calculate average recall

	.. math::
		AR = 2 \times \int_\text{iou_threshold_min}^\text{iou_threshold_max} recall(o)do

	Args:
		iou_threshold_min (float): AR score minimum IoU threshold , default is 0.5
		iou_threshold_max (float): AR score maxiumum IoU threshold , default is 1.0
	"""

	def __init__(self, iou_threshold_min: float = .5,
			iou_threshold_max: float = 1.0):
		super().__init__(
			dist_sync_on_step=False,
			compute_on_step=False)

		self.iou_threshold_min = iou_threshold_min
		self.iou_threshold_max = iou_threshold_max
		self.iou_step = 0.01
		self.add_state("ious", default=[], dist_reduce_fx=None)

	def update(self, preds: List[torch.Tensor], targets: List[torch.Tensor]):
		"""
		Arguments:
			preds {List} -- [Ni,4 dimensional as xmin,ymin,xmax,ymax]
			targets {List} -- [Ni,4 dimensional as xmin,ymin,xmax,ymax]
		"""
		# TODO this might speed down the pipeline

		if not isinstance(preds, List):
			preds = [preds]

		if not isinstance(targets, List):
			targets = [targets]

		ious = []
		for pred, gt in zip(preds, targets):
			# pred: N,4 as xmin, ymin, xmax, ymax
			# gt: M,4 as xmin, ymin, xmax, ymax

			N = pred.size(0)
			M = gt.size(0)

			if M == 0:
				continue

			if N == 0:
				[ious.append(0.0) for _ in range(M)]
				continue

			# N,M
			iou = jaccard_vectorized(pred[:, :4], gt[:, :4])
			select_i, select_j = linear_sum_assignment(-1*iou.numpy())
			select_i = torch.tensor(select_i) # pylint: disable=not-callable
			select_j = torch.tensor(select_j) # pylint: disable=not-callable
			ious += iou[select_i, select_j].tolist()

		# pylint: disable=no-member
		self.ious += ious

	def compute(self):
		"""Calculates average recall
		"""

		# pylint: disable=no-member
		# pylint: disable=not-callable
		ious = torch.tensor(self.ious, dtype=torch.float32)

		recalls = []
		thresholds = torch.arange(start=self.iou_threshold_min,
			end=self.iou_threshold_max, step=self.iou_step)

		for th in thresholds:
			mask = ious >= th
			recalls.append(ious[mask].size(0) / (ious.size(0) + 1e-16))

		recalls = torch.tensor(recalls)
		average_recall = 2*torch.trapz(recalls, thresholds)

		return average_recall