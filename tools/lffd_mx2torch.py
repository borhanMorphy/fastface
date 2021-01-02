import mxnet as mx
import mypackage
from typing import List,Dict
from collections import OrderedDict
import torch
import argparse
import os
import logging

def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-mx-model', '-im',
        type=str, help='mxnet .params model to convert', required=True)

    ap.add_argument('--model-configuration', '-mc', type=str,
        choices=['560_25L_8S'], default='560_25L_8S')

    ap.add_argument('--output-path', '-o', type=str, default='./models')

    return ap.parse_args()

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
    n_head_cls_w = head_cls_w[[0], ...] - head_cls_w[[1], ...]
    return n_head_cls_w

def convert_cls_head_bias_to_binary(head_cls_b:torch.Tensor):
    n_head_cls_b = head_cls_b[[0]] - head_cls_b[[1]]
    return n_head_cls_b

if __name__ == "__main__":
    args = parse_arguments()
    logging.getLogger().setLevel(logging.INFO)

    assert os.path.isdir(args.output_path),f"given {args.output_path} must be directory"
    assert os.path.exists(args.input_mx_model),f"given {args.input_mx_model} mxnet model path does not exists"
    assert args.input_mx_model.endswith(".params"),f"given mxnet model extension must be `.params`"

    logging.info(f"building the lffd architecture with {args.model_configuration} configuration")
    model = mypackage.module.build("lffd", config=args.model_configuration, num_classes=1, in_channels=3)

    logging.info("extracting the state dictionary")
    t_d = model.state_dict()

    logging.info("loading mxnet model")
    mx_d = mx.nd.load(args.input_mx_model)

    t_mx = re_order_mx_keys(mx_d)
    n_st = OrderedDict()

    for k1,k2 in zip(t_d.keys(),t_mx.keys()):
        # converting 2 class classification to binary classification
        if k1.endswith(".cls_head.2.bias"):
            n_st[k1] = convert_cls_head_bias_to_binary(t_mx[k2])
        elif k1.endswith(".cls_head.2.weight"):
            n_st[k1] = convert_cls_head_weight_to_binary(t_mx[k2])
        else:
            n_st[k1] = t_mx[k2]
    model_name = f"original_lffd_{args.model_configuration}.pt"
    model_save_path = os.path.join(args.output_path, model_name)

    if not os.path.exists(args.output_path):
        logging.warning("given output path does not exists, creating...")
        os.makedirs(args.output_path, exist_ok=True)

    torch.save(n_st, model_save_path)