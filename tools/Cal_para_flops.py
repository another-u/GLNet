# import torchsummary
# from torchsummary import summary
# from torchstat import stat
from detectron2.engine.defaults import DefaultPredictor, default_setup, default_argument_parser

from thop import profile,clever_format
import torch
import argparse

from torch.autograd import Variable

from adet.config import get_cfg
from detectron2.modeling import build_model
from collections import OrderedDict

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",
                        default="../GLNet/configs/CIS_P2T.yaml",
                        metavar="FILE", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

args = default_argument_parser().parse_args()
args.config_file = "../GLNet/configs/CIS_P2T_GLNet.yaml"
args.num_gpus = 1
args.confidence_threshold = 0.3
cfg = setup_cfg(args)

model = build_model(cfg)
input = [{'image': torch.zeros(( 3, 800, 800)), 'height': 800, 'width': 800}]

url = r"../GLNet/model_final.pth"
state_dict = torch.load(url)


model.load_state_dict(state_dict, strict=False)
print(model)
total = sum([param.nelement() for param in model.parameters()])
print("total_param:{}".format(total))
print("total_param: %.2fM" % (total/1e6))


device = torch.device('cuda')
model.to(device)
flops, parms = profile(model, inputs=(input,), verbose=False)
flops, parms = clever_format([flops, parms],"%.3f")
print("flops:{}".format(flops), "params:{}".format(parms))

