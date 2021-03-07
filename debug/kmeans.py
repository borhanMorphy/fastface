import torch
import random
import time
from typing import Tuple

class KMeans():
    def __init__(self, k:int, nstart:int=3, seed:bool=False):
        self.nstart = nstart
        self.k = k
        self.seed = seed if nstart == 1 else False

    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)

    def forward(self, boxes:torch.Tensor) -> torch.Tensor:
        # TODO add nstart
        assert boxes.size(1) == 2,"boxes must be form of w,h"

        best_cluster_boxes = torch.empty(self.k, 2, dtype=torch.float32)
        best_avg_iou_score = 0

        if self.seed:
            random.seed(42)

        for _ in range(self.nstart):
            N = boxes.size(0)

            selections = random.sample(range(N), k=self.k)
            cluster_boxes = boxes[selections,:]
            last_clusters = torch.zeros(N).to(boxes.device)

            while True:
                distances = self._distance(boxes, cluster_boxes) # returns N,k distances

                assigned_clusters = torch.argmin(distances, axis=1)

                if (last_clusters == assigned_clusters).all():
                    break

                for i in range(self.k):
                    cluster_boxes[i] = boxes[assigned_clusters==i].median(dim=0)[0]

                last_clusters = assigned_clusters

            avg_iou_score = self.avg_iou(boxes,cluster_boxes)
            if avg_iou_score > best_avg_iou_score:
                best_avg_iou_score = avg_iou_score
                best_cluster_boxes = cluster_boxes
            print("average iou: ",avg_iou_score)
        return best_cluster_boxes,best_avg_iou_score

    def estimate_anchors(self, boxes:torch.Tensor) -> Tuple[torch.Tensor,float]:
        """Estimates anchor sizes with kmeans algoritm
        Arguments:
            boxes {torch.Tensor} -- N,2 dimensional matrix as width,height
        Returns:
            Tuple[torch.Tensor,float] -- normalized_anchors, average iou score
        """
        prior_boxes,avg_iou_score = self.forward(boxes)

        sort = torch.argsort(prior_boxes[:,0] * prior_boxes[:,1])
        return prior_boxes[sort].contiguous(),avg_iou_score

    @staticmethod
    def avg_iou(boxes:torch.Tensor, clusters:torch.Tensor) -> torch.Tensor:
        ious = KMeans._jaccard(boxes, clusters)
        return ious.max(dim=1)[0].mean()

    @staticmethod
    def _distance(boxes:torch.Tensor, clusters:torch.Tensor) -> torch.Tensor:
        # boxes: N,2 => w,h
        # clusters: k,2 => w,h
        return 1-KMeans._jaccard(boxes,clusters)

    @staticmethod
    def _jaccard(box_a:torch.Tensor, box_b:torch.Tensor) -> torch.Tensor:
        inter = KMeans._intersect(box_a,box_b)
        area_a = (box_a[:, 0] * box_a[:, 1]).unsqueeze(1).expand_as(inter).to(box_a.device) # [A,B]
        area_b = (box_b[:, 0] * box_b[:, 1]).unsqueeze(0).expand_as(inter).to(box_b.device) # [A,B]

        union = area_a + area_b - inter
        return inter / union

    @staticmethod
    def _intersect(box_a:torch.Tensor, box_b:torch.Tensor) -> torch.Tensor:
        """
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Args:
            box_a (torch.Tensor): [description]
            box_b (torch.Tensor): [description]
        Returns:
            torch.Tensor: [description]
        """

        A = box_a.size(0)
        B = box_b.size(0)
        min_w = torch.min(box_a[:, :1].unsqueeze(1).expand(A, B, 2),
                          box_b[:, :1].unsqueeze(0).expand(A, B, 2)).to(box_a.device)

        min_h = torch.min(box_a[:, 1:].unsqueeze(1).expand(A, B, 2),
                          box_b[:, 1:].unsqueeze(0).expand(A, B, 2)).to(box_a.device)
        return min_w[:,:,0]*min_h[:,:,0]




if __name__ == '__main__':
    torch.manual_seed(42)
    boxes = torch.rand(800000,2)
    km = KMeans(9)
    st = time.time()
    clusters,prior_boxes = km(boxes)
    print(clusters)
    print(prior_boxes)
    print(f"took: {time.time()-st}")