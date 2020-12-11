import mxnet as mx
from detector import LightFaceDetector
from typing import List,Dict
from collections import OrderedDict
import torch

def re_order_mx_keys(mx_items:Dict):
    b_list = range(25)
    h_list = [8,10,13,15,18,21,23,25]
    # arg:conv{i}_weight
    # arg:conv{i}_bias
    backbone_keys = []
    for b in b_list:
        backbone_keys.append(f"arg:conv{b+1}_weight")
        backbone_keys.append(f"arg:conv{b+1}_bias")
    head_keys = []
    for h in h_list:
        head_keys.append(f"arg:conv{h}_1_weight")
        head_keys.append(f"arg:conv{h}_1_bias")

        head_keys.append(f"arg:conv{h}_2_score_weight")
        head_keys.append(f"arg:conv{h}_2_score_bias")

        head_keys.append(f"arg:conv{h}_3_score_weight")
        head_keys.append(f"arg:conv{h}_3_score_bias")

        head_keys.append(f"arg:conv{h}_2_bbox_weight")
        head_keys.append(f"arg:conv{h}_2_bbox_bias")

        head_keys.append(f"arg:conv{h}_3_bbox_weight")
        head_keys.append(f"arg:conv{h}_3_bbox_bias")
    model_keys = backbone_keys + head_keys
    assert len(model_keys) == len(list(mx_items.keys()))
    st = OrderedDict()
    for k in model_keys:
        st[k] = torch.from_numpy(mx_items[k].asnumpy())
    return st

def convert_cls_head_weight_to_binary(head_cls_w:torch.Tensor):
    print("old head cls weight shape: ",head_cls_w.shape)
    n_head_cls_w = head_cls_w[[0], ...] - head_cls_w[[1], ...]
    print("new head cls weight shape: ",n_head_cls_w.shape)
    print("_"*50)
    return n_head_cls_w

def convert_cls_head_bias_to_binary(head_cls_b:torch.Tensor):
    print("old head cls bias shape: ",head_cls_b.shape)
    n_head_cls_b = head_cls_b[[0]] - head_cls_b[[1]]
    print("new head cls bias shape: ",n_head_cls_b.shape)
    print("_"*50)
    return n_head_cls_b

model = LightFaceDetector.build("lffd")
t_d = model.state_dict()

mx_d = mx.nd.load("/home/morphy/localwork/A-Light-and-Fast-Face-Detector-for-Edge-Devices/face_detection/saved_model/configuration_10_560_25L_8scales_v1/train_10_560_25L_8scales_v1_iter_1400000.params")

t_mx = re_order_mx_keys(mx_d)
n_st = OrderedDict()

for k1,k2 in zip(t_d.keys(),t_mx.keys()):
    #if k1.endswith(".cls_head.2.bias"):
    #    n_st[k1] = convert_cls_head_bias_to_binary(t_mx[k2])
    #elif k1.endswith(".cls_head.2.weight"):
    #    n_st[k1] = convert_cls_head_weight_to_binary(t_mx[k2])
    #else:
    n_st[k1] = t_mx[k2]

torch.save(n_st,"original_lffd.pt")