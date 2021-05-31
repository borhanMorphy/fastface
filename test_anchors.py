import matplotlib.pyplot as plt
import torch
import fastface as ff

selectable_num_anchors = list(range(2,20,2))
num_of_heads = 2

anchor_cov_scores = []
anchors = []

distance_fn = lambda wh_a, wh_b: 1 - ff.utils.box.jaccard_centered(wh_a, wh_b)

ds = ff.dataset.FDDBDataset()

nboxes = ds.get_normalized_boxes()

wh = torch.from_numpy(nboxes[:, [2, 3]] - nboxes[:, [0, 1]])
plt.scatter(wh[:, 0], wh[:, 1])
plt.show()

for i,num_anchors in enumerate(selectable_num_anchors):
    model = ff.utils.cluster.KMeans(num_anchors, distance_fn=distance_fn)
    model.fit(wh)

    # big ones first
    order = (model._params[:, 0] * model._params[:, 1]).argsort(descending=True)
    anchors.append(model._params[order, :].clone())

    ious = ff.utils.box.jaccard_centered(wh, model._params)
    # N,k
    anchor_cov_scores.append(ious.max(dim=1)[0].mean())

    # plt.scatter(model._params[:, 0], model._params[:, 1], c="red")
    # plt.show()

plt.scatter(selectable_num_anchors, anchor_cov_scores, marker="^", c="red")
plt.plot(selectable_num_anchors, anchor_cov_scores, c="blue")
plt.show()

idx = input("select number of anchors: ")

print("best anchors: \n", anchors[int(idx)//num_of_heads-1].reshape(num_of_heads, -1, 2))