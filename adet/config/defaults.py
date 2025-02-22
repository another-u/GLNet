from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = True
_C.INPUT.CROP.CROP_INSTANCE = True

# ---------------------------------------------------------------------------- #
# Basis Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet"
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "SyncBN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 8
_C.MODEL.BASIS_MODULE.NUM_CLASSES = 80
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3

# ---------------------------------------------------------------------------- #
# GLNet Options
# ---------------------------------------------------------------------------- #
_C.MODEL.GLNet = CN()

# Instance hyper-parameters
_C.MODEL.GLNet.INSTANCE_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_C.MODEL.GLNet.FEAT_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
_C.MODEL.GLNet.FEAT_SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
_C.MODEL.GLNet.SIGMA = 0.2
# Channel size for the instance head.
_C.MODEL.GLNet.INSTANCE_IN_CHANNELS = 256
_C.MODEL.GLNet.INSTANCE_CHANNELS = 256
# Convolutions to use in the instance head.
_C.MODEL.GLNet.NUM_INSTANCE_CONVS = 4
_C.MODEL.GLNet.USE_DCN_IN_INSTANCE = False
_C.MODEL.GLNet.TYPE_DCN = 'DCN'
_C.MODEL.GLNet.NUM_GRIDS = [40, 36, 24, 16, 12]
# Number of foreground classes.
_C.MODEL.GLNet.NUM_CLASSES = 80   # COCO
_C.MODEL.GLNet.NUM_KERNELS = 256
_C.MODEL.GLNet.NORM = "GN"
_C.MODEL.GLNet.USE_COORD_CONV = True
_C.MODEL.GLNet.PRIOR_PROB = 0.01

# Mask hyper-parameters.
# Channel size for the mask tower.
_C.MODEL.GLNet.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.GLNet.MASK_IN_CHANNELS = 256
_C.MODEL.GLNet.MASK_CHANNELS = 128
_C.MODEL.GLNet.NUM_MASKS = 256

# Test cfg.
# _C.MODEL.GLNet.CONFIDENCE_SCORE = 0.25
_C.MODEL.GLNet.NMS_PRE = 500
_C.MODEL.GLNet.SCORE_THR = 0.1
_C.MODEL.GLNet.UPDATE_THR = 0.05
_C.MODEL.GLNet.MASK_THR = 0.5
_C.MODEL.GLNet.MAX_PER_IMG = 100
_C.MODEL.GLNet.RESIZE_INPUT_FACTOR = 1
# NMS type: matrix OR mask.
_C.MODEL.GLNet.NMS_TYPE = "matrix"
# Matrix NMS kernel type: gaussian OR linear.
_C.MODEL.GLNet.NMS_KERNEL = "gaussian"
_C.MODEL.GLNet.NMS_SIGMA = 2

# Loss cfg.
_C.MODEL.GLNet.LOSS = CN()
_C.MODEL.GLNet.LOSS.FOCAL_USE_SIGMOID = True
_C.MODEL.GLNet.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.GLNet.LOSS.FOCAL_GAMMA = 2.0
_C.MODEL.GLNet.LOSS.FOCAL_WEIGHT = 1.0
_C.MODEL.GLNet.LOSS.DICE_WEIGHT = 3.0
_C.MODEL.GLNet.LOSS.SEM_WEIGHT = 1.0
_C.MODEL.GLNet.LOSS.INS_EDGE_WEIGHT = 1.0
_C.MODEL.GLNet.LOSS.SEM_TYPE = 'dice'

# Transformer cfg
_C.MODEL.GLNet.HIDDEN_DIM = 256
_C.MODEL.GLNet.NUMBER_FEATURE_LEVELS = 5   # P2 P3 P4 P5 P6
_C.MODEL.GLNet.NHEAD = 8
_C.MODEL.GLNet.ENC_LAYERS = 6
_C.MODEL.GLNet.DEC_LAYERS = 6
_C.MODEL.GLNet.DIM_FEEDFORWARD = 1024
_C.MODEL.GLNet.ENC_POINTS = 4

# Structure cfg
_C.MODEL.GLNet.NOFPN = False
_C.MODEL.GLNet.SEM_LOSS = False
_C.MODEL.GLNet.SINGLE_SEM = False
_C.MODEL.GLNet.INS_EDGE = False
_C.MODEL.GLNet.FFN = 'default'
_C.MODEL.GLNet.INS_FUSION = 'default'
_C.MODEL.GLNet.FSA_IN_NORM = True

# Query selection cfg
_C.MODEL.GLNet.QS = CN()
_C.MODEL.GLNet.QS.ENABLE = True
_C.MODEL.GLNet.QS.INPUT = "ENC"  # ENC/GRID
_C.MODEL.GLNet.QS.SHARE_HEAD = False
_C.MODEL.GLNet.QS.NUM_QUERIES = 300

# ---------------------------------------------------------------------------- #
# PVT Options
# ---------------------------------------------------------------------------- #

_C.MODEL.PVTV2 = CN()
_C.MODEL.PVTV2.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

# ---------------------------------------------------------------------------- #
# SWIN Options
# ---------------------------------------------------------------------------- #

_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.0
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.DROP_RATE = 0.0
_C.MODEL.SWIN.ATTN_DROP_RATE = 0.0
_C.MODEL.SWIN.DROP_PATH_RATE = 0.3
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.SWIN.USE_CHECKPOINT = False


# ---------------------------------------------------------------------------- #
# SOLOv2 Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SOLOV2 = CN()

# Instance hyper-parameters
_C.MODEL.SOLOV2.INSTANCE_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_C.MODEL.SOLOV2.FPN_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
_C.MODEL.SOLOV2.FPN_SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
_C.MODEL.SOLOV2.SIGMA = 0.2
# Channel size for the instance head.
_C.MODEL.SOLOV2.INSTANCE_IN_CHANNELS = 256
_C.MODEL.SOLOV2.INSTANCE_CHANNELS = 512
# Convolutions to use in the instance head.
_C.MODEL.SOLOV2.NUM_INSTANCE_CONVS = 4
_C.MODEL.SOLOV2.USE_DCN_IN_INSTANCE = False
_C.MODEL.SOLOV2.TYPE_DCN = 'DCN'
_C.MODEL.SOLOV2.NUM_GRIDS = [40, 36, 24, 16, 12]
# Number of foreground classes.
_C.MODEL.SOLOV2.NUM_CLASSES = 80
_C.MODEL.SOLOV2.NUM_KERNELS = 256
_C.MODEL.SOLOV2.NORM = "GN"
_C.MODEL.SOLOV2.USE_COORD_CONV = True
_C.MODEL.SOLOV2.PRIOR_PROB = 0.01

# Mask hyper-parameters.
# Channel size for the mask tower.
_C.MODEL.SOLOV2.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.SOLOV2.MASK_IN_CHANNELS = 256
_C.MODEL.SOLOV2.MASK_CHANNELS = 128
_C.MODEL.SOLOV2.NUM_MASKS = 256

# Test cfg.
_C.MODEL.SOLOV2.NMS_PRE = 500
_C.MODEL.SOLOV2.SCORE_THR = 0.1
_C.MODEL.SOLOV2.UPDATE_THR = 0.05
_C.MODEL.SOLOV2.MASK_THR = 0.5
_C.MODEL.SOLOV2.MAX_PER_IMG = 100
# NMS type: matrix OR mask.
_C.MODEL.SOLOV2.NMS_TYPE = "matrix"
# Matrix NMS kernel type: gaussian OR linear.
_C.MODEL.SOLOV2.NMS_KERNEL = "gaussian"
_C.MODEL.SOLOV2.NMS_SIGMA = 2

# Loss cfg.
_C.MODEL.SOLOV2.LOSS = CN()
_C.MODEL.SOLOV2.LOSS.FOCAL_USE_SIGMOID = True
_C.MODEL.SOLOV2.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.SOLOV2.LOSS.FOCAL_GAMMA = 2.0
_C.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT = 1.0
_C.MODEL.SOLOV2.LOSS.DICE_WEIGHT = 3.0
