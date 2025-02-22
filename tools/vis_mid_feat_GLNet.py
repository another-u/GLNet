from detectron2.utils.logger import setup_logger
import torch
from kornia.morphology import erosion

setup_logger()

# import some common libraries
import os, cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor

from adet.config import get_cfg

YML_PATH = '../GLNet/configs/CIS_P2T_GLNet.yaml'
WEIGHTS = '../GLNet/model_final.pth'
OUTPATH = '../GLNet/tools/mid_fea_vis'
if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)


def setup(yml_path, weights):
    cfg = get_cfg()
    cfg.merge_from_file(yml_path)
    cfg.MODEL.WEIGHTS = weights
    # cfg.MODEL.OSFormer.UPDATE_THR = 0.5
    cfg.MODEL.MSPNet.UPDATE_THR = 0.5
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    return cfg, predictor


def vis_features(feat):
    feat = feat.squeeze(0)
    return feat.square().sum(0)


def visualize(im_path, cfg, predictor, out_path):
    im_name = os.path.basename(im_path).split('.')[0]

    model = predictor.model

    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    conv_features = []
    trans_features = []
    camin_features = []
    mask_features = []
    H_W = []

    hooks = [
        # # backbone feature
        # model.backbone.register_forward_hook(
        #     lambda self, input, output: conv_features.append(output["res2"])
        # ),
        # model.backbone.register_forward_hook(
        #     lambda self, input, output: conv_features.append(output["res3"])
        # ),
        # model.backbone.register_forward_hook(
        #     lambda self, input, output: conv_features.append(output["res4"])
        # ),
        # model.backbone.register_forward_hook(
        #     lambda self, input, output: conv_features.append(output["res5"])
        # ),

        # trans feature
        # model.cate_head.trans_encoder.encoder.layers[5].register_forward_hook(
        #     lambda self, input, output: trans_features.append(output)
        # ),

        # mask head feature
        model.mask_head.register_forward_hook(
            lambda self, input, output: mask_features.append(output)
        ),

        # # camin feature
        # model.dcin.register_forward_hook(
        #     lambda self, input, output: camin_features.append(output)
        # )
    ]

    outputs = predictor(im)

    for hook in hooks:
        hook.remove()

    # # save res feats, res2-res5  backbone features
    # spatial_shapes = []
    # spatial_sizes = []
    # for idx, elem in enumerate(conv_features):
    #     cur_feat = vis_features(elem).cpu().detach().numpy()
    #     spatial_shapes.append(tuple(cur_feat.shape))
    #     spatial_sizes.append(cur_feat.shape[0] * cur_feat.shape[1])
    #     plt.axis('off')
    #     plt.imshow(cur_feat)
    #     plt.savefig(os.path.join(out_path, 'vis_res{}_{}.pdf'.format(idx + 2, im_name)), bbox_inches='tight',
    #                 pad_inches=0.0)
    #     print(os.path.join(out_path, 'vis_res{}_{}.pdf'.format(idx + 2, im_name)))

    # # save trans feats, trans3-trans5  PLT Head features
    # for idx, elem, (x, y) in zip(range(len(spatial_shapes) - 1), trans_features[0].split(spatial_sizes[1:], 1),
    #                              spatial_shapes[1:]):
    #     feat = vis_features(elem.permute(0, 2, 1).view(1, -1, x, y)).cpu().numpy()
    #     plt.axis('off')
    #     plt.imshow(feat, cmap='jet')
    #     plt.savefig(os.path.join(out_path, 'vis_trans{}_{}.pdf'.format(idx + 3, im_name)), bbox_inches='tight',
    #                 pad_inches=0.0)
    #     print(os.path.join(out_path, 'vis_trans{}_{}.pdf'.format(idx + 3, im_name)))
    #
    # # save camin output features
    # # camin_features[0] = model.dcin.sigmoid(camin_features[0])
    # camin_feats = camin_features[0].squeeze(0).cpu().numpy()
    # for i in range(camin_feats.shape[0]):
    #     feat = camin_feats[i]
    #     plt.cla()  # ref https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib
    #     plt.axis('off')
    #     plt.imshow(feat, cmap='jet')
    #     # plt.savefig(os.path.join(out_path, 'vis_dcin{}_{}.pdf'.format(i, im_name)), bbox_inches='tight', pad_inches=0.0)
    #     # print(os.path.join(out_path, 'vis_dcin{}_{}.pdf'.format(i, im_name)))
    # plt.savefig(os.path.join(out_path, 'vis_dcin_{}.png'.format(im_name)), bbox_inches='tight', pad_inches=0.0)
    #
    # save mask features
    mask_edge = map_to_edge(mask_features[0][1][0])
    mask_feats = vis_features(mask_edge).cpu().numpy()
    plt.axis('off')
    plt.imshow(mask_feats, cmap='jet')
    plt.savefig(os.path.join(out_path, 'vis_maskhead_edge_E1_{}.jpg'.format(im_name)), bbox_inches='tight', pad_inches=0.0)
    print(os.path.join(out_path, 'vis_maskhead_edge_E1_{}.jpg'.format(im_name)))

    # save rea edges
    # for i in range(len(mask_features[0][1])):
    #     feat = mask_features[0][1][i].squeeze().cpu().numpy()
    #     plt.axis('off')
    #     plt.imshow(feat, cmap='jet')
    #     plt.savefig(os.path.join(out_path, 'vis_rea_edge{}_{}.pdf'.format(i, im_name)), bbox_inches='tight',
    #                 pad_inches=0.0)
    #     print(os.path.join(out_path, 'vis_rea_edge{}_{}.pdf'.format(i, im_name)))

def map_to_edge(tensor):
    tensor = tensor.float()
    kernel = torch.ones((5, 5), device=tensor.device)
    ero_map = erosion(tensor, kernel)
    res = tensor - ero_map

    return res

cfg, predictor = setup(YML_PATH, WEIGHTS)
img_path = '../GLNet/dataset/NC4K/Imgs/1058.jpg'
visualize(img_path, cfg, predictor, OUTPATH)
