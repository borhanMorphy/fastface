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

model = LightFaceDetector.from_pretrained("lffd", "original_lffd.pt")
t_d = model.state_dict()

mx_d = mx.nd.load("/home/morphy/localwork/A-Light-and-Fast-Face-Detector-for-Edge-Devices/face_detection/saved_model/configuration_10_560_25L_8scales_v1/train_10_560_25L_8scales_v1_iter_1400000.params")

t_mx = re_order_mx_keys(mx_d)

for k1,k2 in zip(t_d.keys(),t_mx.keys()):
    check = (t_d[k1] == t_mx[k2]).all()
    print(k1," == ",k2," is ",check)
    assert check,f"{k1} != {k2}"