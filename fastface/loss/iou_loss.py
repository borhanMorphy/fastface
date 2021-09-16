import torch


class DIoULoss(torch.nn.Module):
    """DIoU loss"""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """calculates distance IoU loss

        Args:
            input (torch.Tensor): N,4 as xmin, ymin, xmax, ymax
            target (torch.Tensor): N,4 as xmin, ymin, xmax, ymax

        Returns:
            torch.Tensor: loss
        """
        # TODO do not get mean
        dtype = input.dtype
        device = input.device
        if input.size(0) == 0:
            # pylint: disable=not-callable
            return torch.tensor(0, dtype=dtype, device=device, requires_grad=True)

        eps = 1e-16

        # intersection area
        intersection = (
            # min_x2 - max_x1
            (
                torch.min(input[:, 2], target[:, 2])
                - torch.max(input[:, 0], target[:, 0])
            ).clamp(0)
            *
            # min_y2 - max_y1
            (
                torch.min(input[:, 3], target[:, 3])
                - torch.max(input[:, 1], target[:, 1])
            ).clamp(0)
        )

        # union area
        input_wh = input[:, [2, 3]] - input[:, [0, 1]]
        target_wh = target[:, [2, 3]] - target[:, [0, 1]]
        union = (
            (input_wh[:, 0] * input_wh[:, 1])
            + (target_wh[:, 0] * target_wh[:, 1])
            - intersection
        )

        IoU = intersection / (union + eps)

        enclosing_box_w = torch.max(input[:, 2], target[:, 2]) - torch.min(
            input[:, 0], target[:, 0]
        )
        enclosing_box_h = torch.max(input[:, 3], target[:, 3]) - torch.min(
            input[:, 1], target[:, 1]
        )

        # convex diagonal squared
        c_square = enclosing_box_w ** 2 + enclosing_box_h ** 2

        # squared euclidian distance between box centers
        input_cx = (input[:, 0] + input[:, 2]) / 2
        input_cy = (input[:, 1] + input[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2

        p_square = (input_cx - target_cx) ** 2 + (input_cy - target_cy) ** 2

        penalty = p_square / (c_square + eps)

        # DIoU loss
        return 1 - IoU + penalty
