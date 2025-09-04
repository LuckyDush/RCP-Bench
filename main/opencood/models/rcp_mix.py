# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# A unified framework for LiDAR-only / Camera-only / Heterogeneous collaboration.
# Support multiple fusion strategies.


import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import MaxFusion, AttFusion, DiscoFusion, V2VNetFusion, V2XViTFusion, CoBEVT, Where2commFusion, Who2comFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision


class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x
        ####x.shape  torch.Size([1, 128, 256, 256])
        if x.shape[0]==1:
            return x
        else:
            mean = x.mean(dim=[2, 3], keepdim=False)  ####torch.Size([1, 128])
            std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()  ####torch.Size([1, 128])

            sqrtvar_mu = self.sqrtvar(mean) ####torch.Size([1, 128])
            sqrtvar_std = self.sqrtvar(std)

            beta = self._reparameterize(mean, sqrtvar_mu)
            gamma = self._reparameterize(std, sqrtvar_std)

            x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
            x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

            return x

def calc_mean_std(feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
def adaptive_instance_normalization(all_cv_feat):
    size = all_cv_feat[0].unsqueeze(0).size()
    allcv_mean1 , allcv_std1 = calc_mean_std(all_cv_feat)
    allcv_mean=torch.mean(allcv_mean1,dim=0).unsqueeze(0)
    allcv_std=torch.mean(allcv_std1,dim=0).unsqueeze(0)
    beta = torch.distributions.Beta(0.1, 0.1)
    youhua = []
    for i in range(all_cv_feat.shape[0]):
        if i == 0:
            ego_mean, ego_std = calc_mean_std(all_cv_feat[i].unsqueeze(0))
        #     std=0.1*ego_std+allcv_std*0.9
        #     mean=0.1*ego_mean+allcv_mean*0.9
        #     normalized_ego_feat = (all_cv_feat[i] - ego_mean.expand(
        # size)) / ego_std.expand(size)
        #     egoselfyouhua_feat= normalized_ego_feat * std.expand(size) + mean.expand(size)
            youhua.append(all_cv_feat[i].unsqueeze(0))
        else:
            locals()[f'other{i}_mean'], locals()[f'other{i}_std'] = calc_mean_std(all_cv_feat[i].unsqueeze(0))
            locals()[f'normalized{i}_feat'] = (all_cv_feat[i].unsqueeze(0) - locals()[f'other{i}_mean'].expand(
        size)) /locals()[f'other{i}_std'].expand(size)
            std=0.1*locals()[f'other{i}_std']+allcv_std*0.9
            mean=0.1*locals()[f'other{i}_mean']+allcv_mean*0.9
            locals()[f'egoyouhua{i}_feat']=locals()[f'normalized{i}_feat'] * std.expand(size) + mean.expand(size)
            youhua.append(locals()[f'egoyouhua{i}_feat'])
    return  torch.stack(youhua).squeeze(1)

def adaptive_instance_normalizationallcvinclego(all_cv_feat):
    size = all_cv_feat[0].unsqueeze(0).size()
    allcv_mean1 , allcv_std1 = calc_mean_std(all_cv_feat)
    allcv_mean=torch.mean(allcv_mean1,dim=0).unsqueeze(0)
    allcv_std=torch.mean(allcv_std1,dim=0).unsqueeze(0)
    beta = torch.distributions.Beta(0.1, 0.1)
    youhua = []
    for i in range(all_cv_feat.shape[0]):
        if i == 0:
            ego_mean, ego_std = calc_mean_std(all_cv_feat[i].unsqueeze(0))
            normalized_ego_feat = (all_cv_feat[i].unsqueeze(0) - ego_mean.expand(
        size)) / ego_std.expand(size)
            lmda = beta.sample()
            std=lmda*ego_std+allcv_std*(1-lmda)
            mean=lmda*ego_mean+allcv_mean*(1-lmda)
            egoselfyouhua_feat= normalized_ego_feat * std.expand(size) + mean.expand(size)
            youhua.append(egoselfyouhua_feat)
        else:
            locals()[f'other{i}_mean'], locals()[f'other{i}_std'] = calc_mean_std(all_cv_feat[i].unsqueeze(0))
            locals()[f'normalized{i}_feat'] = (all_cv_feat[i].unsqueeze(0) - locals()[f'other{i}_mean'].expand(
        size)) /locals()[f'other{i}_std'].expand(size)
            lmda = beta.sample()
            std=lmda*locals()[f'other{i}_std']+allcv_std*(1-lmda)
            mean=lmda*locals()[f'other{i}_mean']+allcv_mean*(1-lmda)
            locals()[f'allcvyouhua{i}_feat']=locals()[f'normalized{i}_feat'] * std.expand(size) + mean.expand(size)
            youhua.append(locals()[f'allcvyouhua{i}_feat'])
    return  torch.stack(youhua).squeeze(1)

def adaptive_instance_normalizationallcvinclegotest(all_cv_feat):
    size = all_cv_feat[0].unsqueeze(0).size()
    allcv_mean1 , allcv_std1 = calc_mean_std(all_cv_feat)
    allcv_mean=torch.mean(allcv_mean1,dim=0).unsqueeze(0)
    allcv_std=torch.mean(allcv_std1,dim=0).unsqueeze(0)
    beta = torch.distributions.Beta(0.1, 0.1)
    youhua = []
    for i in range(all_cv_feat.shape[0]):
        if i == 0:
            ego_mean, ego_std = calc_mean_std(all_cv_feat[i].unsqueeze(0))
            normalized_ego_feat = (all_cv_feat[i].unsqueeze(0) - ego_mean.expand(
        size)) / ego_std.expand(size)
            lmda = 0.5
            std=lmda*ego_std+allcv_std*(1-lmda)
            mean=lmda*ego_mean+allcv_mean*(1-lmda)
            egoselfyouhua_feat= normalized_ego_feat * std.expand(size) + mean.expand(size)
            youhua.append(egoselfyouhua_feat)
        else:
            locals()[f'other{i}_mean'], locals()[f'other{i}_std'] = calc_mean_std(all_cv_feat[i].unsqueeze(0))
            locals()[f'normalized{i}_feat'] = (all_cv_feat[i].unsqueeze(0) - locals()[f'other{i}_mean'].expand(
        size)) /locals()[f'other{i}_std'].expand(size)
            lmda = 0.5
            std=lmda*locals()[f'other{i}_std']+allcv_std*(1-lmda)
            mean=lmda*locals()[f'other{i}_mean']+allcv_mean*(1-lmda)
            locals()[f'allcvyouhua{i}_feat']=locals()[f'normalized{i}_feat'] * std.expand(size) + mean.expand(size)
            youhua.append(locals()[f'allcvyouhua{i}_feat'])
    return  torch.stack(youhua).squeeze(1)

def adaptive_instance_normalizationallcv(all_cv_feat):
    size = all_cv_feat[0].unsqueeze(0).size()
    allcv_mean1 , allcv_std1 = calc_mean_std(all_cv_feat)
    allcv_mean=torch.mean(allcv_mean1,dim=0).unsqueeze(0)
    allcv_std=torch.mean(allcv_std1,dim=0).unsqueeze(0)
    beta = torch.distributions.Beta(0.1, 0.1)
    youhua = []
    for i in range(all_cv_feat.shape[0]):
        if i == 0:
            ego_mean, ego_std = calc_mean_std(all_cv_feat[i].unsqueeze(0))
            normalized_ego_feat = (all_cv_feat[i] - ego_mean.expand(
        size)) / ego_std.expand(size)
            youhua.append(all_cv_feat[i].unsqueeze(0))
        else:
            locals()[f'other{i}_mean'], locals()[f'other{i}_std'] = calc_mean_std(all_cv_feat[i].unsqueeze(0))
            locals()[f'normalized{i}_feat'] = (all_cv_feat[i] - locals()[f'other{i}_mean'].expand(
        size)) /locals()[f'other{i}_std'].expand(size)
            lmda = beta.sample()
            std=lmda*locals()[f'other{i}_std']+allcv_std*(1-lmda)
            mean=lmda*locals()[f'other{i}_mean']+allcv_mean*(1-lmda)
            locals()[f'allcvyouhua{i}_feat']=locals()[f'normalized{i}_feat'] * std.expand(size) + mean.expand(size)
            youhua.append(locals()[f'allcvyouhua{i}_feat'])
    return  torch.stack(youhua).squeeze(1)

def adaptive_instance_normalizationallcvtest(all_cv_feat):
    size = all_cv_feat[0].unsqueeze(0).size()
    allcv_mean1 , allcv_std1 = calc_mean_std(all_cv_feat)
    allcv_mean=torch.mean(allcv_mean1,dim=0).unsqueeze(0)
    allcv_std=torch.mean(allcv_std1,dim=0).unsqueeze(0)
    beta = torch.distributions.Beta(0.1, 0.1)
    youhua = []
    for i in range(all_cv_feat.shape[0]):
        if i == 0:
            ego_mean, ego_std = calc_mean_std(all_cv_feat[i].unsqueeze(0))
            normalized_ego_feat = (all_cv_feat[i] - ego_mean.expand(
        size)) / ego_std.expand(size)
            youhua.append(all_cv_feat[i].unsqueeze(0))
        else:
            locals()[f'other{i}_mean'], locals()[f'other{i}_std'] = calc_mean_std(all_cv_feat[i].unsqueeze(0))
            locals()[f'normalized{i}_feat'] = (all_cv_feat[i] - locals()[f'other{i}_mean'].expand(
        size)) /locals()[f'other{i}_std'].expand(size)
            lmda = 0.5
            std=lmda*locals()[f'other{i}_std']+allcv_std*(1-lmda)
            mean=lmda*locals()[f'other{i}_mean']+allcv_mean*(1-lmda)
            locals()[f'allcvyouhua{i}_feat']=locals()[f'normalized{i}_feat'] * std.expand(size) + mean.expand(size)
            youhua.append(locals()[f'allcvyouhua{i}_feat'])
    return  torch.stack(youhua).squeeze(1)

class HeterModelBaseline(nn.Module):
    def __init__(self, args):
        super(HeterModelBaseline, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()]
        self.modality_name_list = modality_name_list
        self.uncertainty_module = DistributionUncertainty(p=0.5)
        self.ego_modality = args['ego_modality']

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building
            """
            setattr(self, f"backbone_{modality_name}", BaseBEVBackbone(model_setting['backbone_args'],
                                                                       model_setting['backbone_args'].get('inplanes',64)))

            """
            shrink conv building
            """
            setattr(self, f"shrinker_{modality_name}", DownsampleConv(model_setting['shrink_header']))

            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))

        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        self.supervise_single = False
        if args.get("supervise_single", False):
            self.supervise_single = True
            in_head_single = args['in_head_single']
            setattr(self, f'cls_head_single', nn.Conv2d(in_head_single, args['anchor_number'], kernel_size=1))
            setattr(self, f'reg_head_single', nn.Conv2d(in_head_single, args['anchor_number'] * 7, kernel_size=1))
            setattr(self, f'dir_head_single', nn.Conv2d(in_head_single, args['anchor_number'] *  args['dir_args']['num_bins'], kernel_size=1))


        if args['fusion_method'] == "max":
            self.fusion_net = MaxFusion()
        if args['fusion_method'] == "att":
            self.fusion_net = AttFusion(args['att']['feat_dim'])
        if args['fusion_method'] == "disconet":
            self.fusion_net = DiscoFusion(args['disconet']['feat_dim'])
        if args['fusion_method'] == "v2vnet":
            self.fusion_net = V2VNetFusion(args['v2vnet'])
        if args['fusion_method'] == 'v2xvit':
            self.fusion_net = V2XViTFusion(args['v2xvit'])
        if args['fusion_method'] == 'cobevt':
            self.fusion_net = CoBEVT(args['cobevt'])
        if args['fusion_method'] == 'where2comm':
            self.fusion_net = Where2commFusion(args['where2comm'])
        if args['fusion_method'] == 'who2com':
            self.fusion_net = Who2comFusion(args['who2com'])


        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])
            self.model_train_init()


        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self):
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def forward(self, data_dict):
        output_dict = {}
        agent_modality_list = data_dict['agent_modality_list']
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len']
        # print(agent_modality_list)

        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            # import pdb
            # pdb.set_trace()
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature=adaptive_instance_normalizationallcvinclego(feature)
            # feature=adaptive_instance_normalizationallcvinclegotest(feature)
            ###torch.Size([2, 128, 256, 256])
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            # feature=adaptive_instance_normalizationallcvinclego(feature)
            # feature=adaptive_instance_normalizationallcvinclegotest(feature)
            ####torch.Size([2, 384, 128, 128])
            feature = eval(f"self.shrinker_{modality_name}")(feature)
            feature=adaptive_instance_normalizationallcvinclego(feature)
            # feature=adaptive_instance_normalizationallcvinclegotest(feature)
            #####torch.Size([2, 256, 128, 128])
            modality_feature_dict[modality_name] = feature


        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        """
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        """
        Single supervision
        """
        if self.supervise_single:
            cls_preds_before_fusion = self.cls_head_single(heter_feature_2d)
            reg_preds_before_fusion = self.reg_head_single(heter_feature_2d)
            dir_preds_before_fusion = self.dir_head_single(heter_feature_2d)
            output_dict.update({'cls_preds_single': cls_preds_before_fusion,
                                'reg_preds_single': reg_preds_before_fusion,
                                'dir_preds_single': dir_preds_before_fusion})

        """
        Feature Fusion (multiscale).

        we omit self.backbone's first layer.
        """
        fused_feature = self.fusion_net(heter_feature_2d, record_len, affine_matrix)

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        cls_preds = self.cls_head(fused_feature)
        reg_preds = self.reg_head(fused_feature)
        dir_preds = self.dir_head(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})

        return output_dict
