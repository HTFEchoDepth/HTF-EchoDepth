import torch
import torch.nn as nn
from torch.nn import init
import functools

from htf_echodepth.models.backbone.fadc_inspired_v0 import FreqRFBlockV0
from htf_echodepth.models.backbone.fadc_inspired_v1a import FreqRFBlockV1a, reapply_freqselect_v1a_zero_init
from htf_echodepth.models.backbone.fadc_inspired_full_single import FreqRFBlockFullSingle, reapply_fadc_full_single_zero_init
from htf_echodepth.models.backbone.transformer_bottleneck import TransformerBottleneckT8, reapply_transformer_bottleneck_gamma_zero

# UNet based on Cycle GAN pytorch implementation: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        first = gpu_ids[0]
        if isinstance(first, torch.device) and first.type == "cpu":
            net.to(first)
        else:
            assert torch.cuda.is_available()
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class Identity(nn.Module):
    def forward(self, x):
        return x


def define_G(cfg, input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'ahmf_former_hybrid_v1_net':
        from htf_echodepth.models.backbone.ahmf_former_hybrid_v1_net import define_ahmf_former_hybrid_v1
        return define_ahmf_former_hybrid_v1(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'ahmf_former_hybrid_sgc_net':
        from htf_echodepth.models.backbone.ahmf_former_hybrid_sgc_net import define_ahmf_former_hybrid_sgc
        return define_ahmf_former_hybrid_sgc(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'ahmf_former_hybrid_net':
        from htf_echodepth.models.backbone.ahmf_former_hybrid_net import define_ahmf_former_hybrid
        return define_ahmf_former_hybrid(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'ahmf_former_net':
        from htf_echodepth.models.backbone.ahmf_former_net import define_ahmf_former
        return define_ahmf_former(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'esf_refiner_v0_net':
        from htf_echodepth.models.backbone.esf_refiner_v0 import define_esf_refiner_v0
        return define_esf_refiner_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'herd_depth_v0_net':
        from htf_echodepth.models.backbone.herd_depth_v0 import define_herd_depth_v0
        return define_herd_depth_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'herd_depth_v0_1_net':
        from htf_echodepth.models.backbone.herd_depth_v0_1 import define_herd_depth_v0_1
        return define_herd_depth_v0_1(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'herd_joint_v0_net':
        from htf_echodepth.models.backbone.herd_joint_v0 import define_herd_joint_v0
        return define_herd_joint_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'herd_depth_v0_2_c1_net':
        from htf_echodepth.models.backbone.herd_depth_v0_2_c1 import define_herd_depth_v0_2_c1
        return define_herd_depth_v0_2_c1(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'herd_depth_v0_2_c1_1_net':
        from htf_echodepth.models.backbone.herd_depth_v0_2_c1_1 import define_herd_depth_v0_2_c1_1
        return define_herd_depth_v0_2_c1_1(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'herd_depth_v0_2_c2_net':
        from htf_echodepth.models.backbone.herd_depth_v0_2_c2 import define_herd_depth_v0_2_c2
        return define_herd_depth_v0_2_c2(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'ahmf_net':
        from htf_echodepth.models.backbone.ahmf_net import define_ahmf
        return define_ahmf(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'tf_decoupled_ahmf_encoder_v0_net':
        from htf_echodepth.models.backbone.tf_decoupled_ahmf_encoder_v0_net import define_tf_decoupled_ahmf_encoder_v0
        return define_tf_decoupled_ahmf_encoder_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'tfms_block_encoder_v1_net':
        from htf_echodepth.models.backbone.tfms_block_encoder_v1_net import define_tfms_block_encoder_v1
        return define_tfms_block_encoder_v1(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'tfms_block_encoder_v1_1_net':
        from htf_echodepth.models.backbone.tfms_block_encoder_v1_1_net import define_tfms_block_encoder_v1_1
        return define_tfms_block_encoder_v1_1(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'tfms_block_encoder_v2_net':
        from htf_echodepth.models.backbone.tfms_block_encoder_v2_net import define_tfms_block_encoder_v2
        return define_tfms_block_encoder_v2(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'tfms_dynamic_alpha_v0_net':
        from htf_echodepth.models.backbone.tfms_dynamic_alpha_v0_net import define_tfms_dynamic_alpha_v0
        return define_tfms_dynamic_alpha_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'tfms_selective_no_harm_v0_net':
        from htf_echodepth.models.backbone.tfms_selective_no_harm_v0 import define_tfms_selective_no_harm_v0
        return define_tfms_selective_no_harm_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'fn_ahmf_bridge_v0_net':
        from htf_echodepth.models.backbone.fn_ahmf_bridge_v0 import define_fn_ahmf_bridge_v0
        return define_fn_ahmf_bridge_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_unet5_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v0 import define_res_tfc_tdf_unet5_v0
        return define_res_tfc_tdf_unet5_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_unet5_v1_aligned':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned import define_res_tfc_tdf_unet5_v1_aligned
        return define_res_tfc_tdf_unet5_v1_aligned(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'resconv_unet5_v0':
        from htf_echodepth.models.backbone.resconv_unet5_v0 import define_resconv_unet5_v0
        return define_resconv_unet5_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'resconv_imafenc_unet5_v0':
        from htf_echodepth.models.backbone.resconv_imafenc_unet5_v0 import define_resconv_imafenc_unet5_v0
        return define_resconv_imafenc_unet5_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'resconv_dp_unet5_v0':
        from htf_echodepth.models.backbone.resconv_dp_unet5_v0 import define_resconv_dp_unet5_v0
        return define_resconv_dp_unet5_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'batvision_style_unet5_l1_v0':
        from htf_echodepth.models.backbone.batvision_style_unet5_l1_v0 import define_batvision_style_unet5_l1_v0
        return define_batvision_style_unet5_l1_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'batvision_style_imafenc_unet5_v0':
        from htf_echodepth.models.backbone.batvision_style_imafenc_unet5_v0 import define_batvision_style_imafenc_unet5_v0
        return define_batvision_style_imafenc_unet5_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'echonet_audioonly_full_zero_rgb_bv2_v0':
        from htf_echodepth.models.backbone.echonet_audioonly_full_zero_rgb_bv2_v0 import define_echonet_audioonly_full_zero_rgb_bv2_v0
        return define_echonet_audioonly_full_zero_rgb_bv2_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'echonet_simple_audio_depthnet_only_bv2_v0':
        from htf_echodepth.models.backbone.echonet_simple_audio_depthnet_only_bv2_v0 import define_echonet_simple_audio_depthnet_only_bv2_v0
        return define_echonet_simple_audio_depthnet_only_bv2_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_fa_dtf_tdf_allstage_imafenc_unet5_l1_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_allstage_imafenc_unet5_l1_v0 import (
            define_res_fa_dtf_tdf_allstage_imafenc_unet5_l1_v0,
        )
        return define_res_fa_dtf_tdf_allstage_imafenc_unet5_l1_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_allstage_imafenc_dualpath_unet5_l1_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_allstage_imafenc_dualpath_unet5_l1_v0 import (
            define_res_fa_dtf_tdf_allstage_imafenc_dualpath_unet5_l1_v0,
        )
        return define_res_fa_dtf_tdf_allstage_imafenc_dualpath_unet5_l1_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_dtfconv_unet5_v1_aligned_v0':
        from htf_echodepth.models.backbone.res_dtfconv_unet5_v1_aligned_v0 import define_res_dtfconv_unet5_v1_aligned_v0
        return define_res_dtfconv_unet5_v1_aligned_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_dtf_mbconv_unet5_v1_aligned_v0':
        from htf_echodepth.models.backbone.res_dtf_mbconv_unet5_v1_aligned_v0 import define_res_dtf_mbconv_unet5_v1_aligned_v0
        return define_res_dtf_mbconv_unet5_v1_aligned_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_dtf_tdf_v3_v0_unet5_v1_aligned_v0':
        from htf_echodepth.models.backbone.res_dtf_tdf_v3_v0_unet5_v1_aligned_v0 import define_res_dtf_tdf_v3_v0_unet5_v1_aligned_v0
        return define_res_dtf_tdf_v3_v0_unet5_v1_aligned_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_fa_dtf_tdf_v3_v1_unet5_v1_aligned_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_v3_v1_unet5_v1_aligned_v0 import define_res_fa_dtf_tdf_v3_v1_unet5_v1_aligned_v0
        return define_res_fa_dtf_tdf_v3_v1_unet5_v1_aligned_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_fa_dtf_notdf_v0_unet5_v1_aligned_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_notdf_v0_unet5_v1_aligned_v0 import (
            define_res_fa_dtf_notdf_v0_unet5_v1_aligned_v0,
        )
        return define_res_fa_dtf_notdf_v0_unet5_v1_aligned_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG.startswith('res_fa_dtf_tdf_dp_v3_v2_'):
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_v3_v2_unet5_v1_aligned_v0 import define_res_fa_dtf_tdf_dp_v3_v2_unet5_v1_aligned_v0
        return define_res_fa_dtf_tdf_dp_v3_v2_unet5_v1_aligned_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0 import (
            define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0,
        )
        return define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_fd_dtf_matrix_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_fd_dtf_matrix_v0 import (
            define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_fd_dtf_matrix_v0,
        )
        return define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_fd_dtf_matrix_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_imaf_dp_notdf_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_imaf_dp_notdf_v0 import define_res_fa_dtf_imaf_dp_notdf_v0
        return define_res_fa_dtf_imaf_dp_notdf_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_afpnr_residualskip_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_afpnr_residualskip_v0 import (
            define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_afpnr_residualskip_v0,
        )
        return define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_afpnr_residualskip_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_denseaspp_middeep_fullgc_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_denseaspp_middeep_fullgc_v0 import (
            define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_denseaspp_middeep_fullgc_v0,
        )
        return define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_denseaspp_middeep_fullgc_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_aligned_eca_aspp_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_aligned_eca_aspp_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_aligned_eca_aspp_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_aligned_eca_aspp_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_aligned_eca_denseaspp_middeep_fullgc_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_aligned_eca_denseaspp_middeep_fullgc_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_aligned_eca_denseaspp_middeep_fullgc_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_aligned_eca_denseaspp_middeep_fullgc_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_serial_resdtf_v1':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_serial_resdtf_v1 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_serial_resdtf_v1,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_serial_resdtf_v1(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_branch_deep_dtf_v2':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_branch_deep_dtf_v2 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_branch_deep_dtf_v2,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_branch_deep_dtf_v2(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_branch_deep_eca_v3':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_branch_deep_eca_v3 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_branch_deep_eca_v3,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_branch_deep_eca_v3(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_fusion_eca_v4':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_fusion_eca_v4 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_fusion_eca_v4,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_fusion_eca_v4(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_dtfpost_eca_aspp_inline_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_dtfpost_eca_aspp_inline_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtfpost_eca_aspp_inline_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtfpost_eca_aspp_inline_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_dtfpost_eca_aspp_inline_a_feature_imdf_residual_skip_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_dtfpost_eca_aspp_inline_a_feature_imdf_residual_skip_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtfpost_eca_aspp_inline_a_feature_imdf_residual_skip_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtfpost_eca_aspp_inline_a_feature_imdf_residual_skip_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_imdf_residual_skip_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_imdf_residual_skip_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_imdf_residual_skip_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_imdf_residual_skip_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_imdf_residual_skip_variant_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_imdf_residual_skip_variant_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_imdf_residual_skip_variant_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_imdf_residual_skip_variant_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_direct_raw_imdf_fullsdi_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_direct_raw_imdf_fullsdi_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_direct_raw_imdf_fullsdi_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_direct_raw_imdf_fullsdi_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_dtfpost_eca_aspp_inline_direct_raw_imdf_fullsdi_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_dtfpost_eca_aspp_inline_direct_raw_imdf_fullsdi_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtfpost_eca_aspp_inline_direct_raw_imdf_fullsdi_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtfpost_eca_aspp_inline_direct_raw_imdf_fullsdi_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_gatedskip_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_gatedskip_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_gatedskip_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_gatedskip_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_deepctxdec_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_deepctxdec_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_deepctxdec_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_deepctxdec_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_hc2r_skipclean_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_hc2r_skipclean_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_hc2r_skipclean_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_hc2r_skipclean_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_bifpn_lite_234post_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_bifpn_lite_234post_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_bifpn_lite_234post_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_bifpn_lite_234post_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_bifpn_residual_skip_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_bifpn_residual_skip_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_bifpn_residual_skip_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_bifpn_residual_skip_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_p0_bounded_small_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_p0_bounded_small_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_p0_bounded_small_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_p0_bounded_small_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_fsp_rmsalign_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_fsp_rmsalign_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_fsp_rmsalign_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_fsp_rmsalign_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_middeep_initboost_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_middeep_initboost_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_middeep_initboost_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_dtf_feature_afpn_residual_skip_middeep_initboost_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_afpn_channel_lambda_residual_skip_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_afpn_channel_lambda_residual_skip_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_afpn_channel_lambda_residual_skip_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_afpn_channel_lambda_residual_skip_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_bmamba_v3maf_afpn_alpha_ca_skip_replace_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_bmamba_v3maf_afpn_alpha_ca_skip_replace_v0 import (
            define_res_fa_dtf_tdf_dp_bmamba_v3maf_afpn_alpha_ca_skip_replace_v0,
        )
        return define_res_fa_dtf_tdf_dp_bmamba_v3maf_afpn_alpha_ca_skip_replace_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_imaf_enc_v3_a_afpn_bv_imdf_lite_safe_notop_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_imaf_enc_v3_a_afpn_bv_imdf_lite_safe_notop_v0 import (
            define_res_fa_dtf_tdf_dp_imaf_enc_v3_a_afpn_bv_imdf_lite_safe_notop_v0,
        )
        return define_res_fa_dtf_tdf_dp_imaf_enc_v3_a_afpn_bv_imdf_lite_safe_notop_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG in (
        'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_gamma_u02_unet5_v1_aligned_v0',
        'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_gamma_sds_unet5_v1_aligned_v0',
    ):
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0 import (
            define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0,
        )
        return define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_a_unet5_v1_aligned_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_b_unet5_v1_aligned_v0':
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_b_unet5_v1_aligned_v0 import (
            define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_b_unet5_v1_aligned_v0,
        )
        return define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_b_unet5_v1_aligned_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG in (
        'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_stage_mask_unet5_v1_aligned_v0',
        'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_s1_mid_only_unet5_v1_aligned_v0',
        'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_s2_no_shallow_unet5_v1_aligned_v0',
        'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_s3_deep_only_unet5_v1_aligned_v0',
        'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_cfinal_s3calib_unet5_v1_aligned_v0',
        'res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_cfinal_wenc2_s3calib_unet5_v1_aligned_v0',
    ):
        from htf_echodepth.models.backbone.res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_stage_mask_unet5_v1_aligned_v0 import (
            define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_stage_mask_unet5_v1_aligned_v0,
        )
        return define_res_fa_dtf_tdf_dp_imaf_enc_v3_v3_h4_l1_stage_mask_unet5_v1_aligned_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_mbconv_control_unet5_v1_aligned_v0':
        from htf_echodepth.models.backbone.res_mbconv_control_unet5_v1_aligned_v0 import define_res_mbconv_control_unet5_v1_aligned_v0
        return define_res_mbconv_control_unet5_v1_aligned_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_imdf_unet5_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_imdf_unet5_v0 import define_res_tfc_tdf_imdf_unet5_v0
        return define_res_tfc_tdf_imdf_unet5_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_imdf_unet5_alpha_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_imdf_unet5_alpha_v0 import define_res_tfc_tdf_imdf_unet5_alpha_v0
        return define_res_tfc_tdf_imdf_unet5_alpha_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_imdf_unet5_v1_aligned_aapn_v0_1':
        from htf_echodepth.models.backbone.res_tfc_tdf_imdf_unet5_v1_aligned_aapn_v0_1 import define_res_tfc_tdf_imdf_unet5_v1_aligned_aapn_v0_1
        return define_res_tfc_tdf_imdf_unet5_v1_aligned_aapn_v0_1(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_v0 import define_res_tfc_tdf_unet5_v1_aligned_imaf_v0
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_paperfull_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_paperfull_v0 import define_res_tfc_tdf_unet5_v1_aligned_imaf_paperfull_v0
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_paperfull_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_v0 import define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_v0
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_fpn_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_fpn_v0 import define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_fpn_v0
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_fpn_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_full_route_fa_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_full_route_fa_v0 import (
            define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_full_route_fa_v0,
        )
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_full_route_fa_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_bv_residual_inject_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_bv_residual_inject_v0 import (
            define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_bv_residual_inject_v0,
        )
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_bv_residual_inject_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_middeep_residual_inject_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_middeep_residual_inject_v0 import (
            define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_middeep_residual_inject_v0,
        )
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_middeep_residual_inject_v0(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_bv_residual_inject_v0_1_rms_calib':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_bv_residual_inject_v0_1_rms_calib import (
            define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_bv_residual_inject_v0_1_rms_calib,
        )
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_bv_residual_inject_v0_1_rms_calib(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_bv_residual_inject_v0_2_s2_rmsonly':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_bv_residual_inject_v0_2_s2_rmsonly import (
            define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_bv_residual_inject_v0_2_s2_rmsonly,
        )
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_asff_bv_residual_inject_v0_2_s2_rmsonly(
            cfg, input_nc, output_nc, init_type, init_gain, gpu_ids
        )
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_afpn_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_afpn_v0 import define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_afpn_v0
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_afpn_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_afpn_bv_adapt_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_afpn_bv_adapt_v0 import define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_afpn_bv_adapt_v0
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_afpn_bv_adapt_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_afpn_bv_imdf_lite_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_afpn_bv_imdf_lite_v0 import define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_afpn_bv_imdf_lite_v0
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_afpn_bv_imdf_lite_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_original_imdf_ebank_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_original_imdf_ebank_v0 import define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_original_imdf_ebank_v0
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_original_imdf_ebank_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_scale_norm_imdf_ebank_v0':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_scale_norm_imdf_ebank_v0 import define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_scale_norm_imdf_ebank_v0
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_scale_norm_imdf_ebank_v0(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_scale_norm_imdf_ebank_v0_1_gamma_calib':
        from htf_echodepth.models.backbone.res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_scale_norm_imdf_ebank_v0_1_gamma_calib import define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_scale_norm_imdf_ebank_v0_1_gamma_calib
        return define_res_tfc_tdf_unet5_v1_aligned_imaf_hybrid_scale_norm_imdf_ebank_v0_1_gamma_calib(cfg, input_nc, output_nc, init_type, init_gain, gpu_ids)
    if netG == 'unet_128':
        net = UnetGenerator(cfg, input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(cfg, input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    net = init_net(net, init_type, init_gain, gpu_ids)
    if bool(getattr(cfg.model, "use_freq_rf_v1a", False)):
        reapply_freqselect_v1a_zero_init(net)
    if bool(getattr(cfg.model, "use_freq_rf_full_single", False)):
        reapply_fadc_full_single_zero_init(net)
    if bool(getattr(cfg.model, "use_transformer_t8", False)):
        reapply_transformer_bottleneck_gamma_zero(net)
    return net


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, cfg, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # FreqRF v0 / v1a: only valid for unet_256 (num_downs==8); 3rd down -> (B, ngf*4, 32, 32) before deeper encoder.
        use_freq_rf_v0 = bool(getattr(cfg.model, "use_freq_rf_v0", False)) and num_downs == 8
        use_freq_rf_v1a = bool(getattr(cfg.model, "use_freq_rf_v1a", False)) and num_downs == 8
        use_freq_rf_full_single = bool(getattr(cfg.model, "use_freq_rf_full_single", False)) and num_downs == 8
        use_transformer_t8 = bool(getattr(cfg.model, "use_transformer_t8", False)) and num_downs == 8
        fadc_flags = (use_freq_rf_v0, use_freq_rf_v1a, use_freq_rf_full_single)
        if sum(fadc_flags) > 1:
            raise ValueError(
                "At most one of use_freq_rf_v0, use_freq_rf_v1a, use_freq_rf_full_single may be true."
            )
        if use_transformer_t8 and any(fadc_flags):
            raise ValueError("use_transformer_t8 cannot be enabled together with FADC modules.")
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(cfg,ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        n_intermediate = num_downs - 5
        for i in range(n_intermediate):          # add intermediate layers with ngf * 8 filters
            # T8 @ 512@8x8: outermost of the ngf*8 stack (16->8 down); i == n_intermediate - 1
            inject_transformer_t8 = use_transformer_t8 and (i == n_intermediate - 1)
            unet_block = UnetSkipConnectionBlock(
                cfg, ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer,
                use_dropout=use_dropout, inject_transformer_t8=inject_transformer_t8,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(cfg,ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            cfg,
            ngf * 2,
            ngf * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            inject_freq_rf_v0=use_freq_rf_v0,
            inject_freq_rf_v1a=use_freq_rf_v1a,
            inject_freq_rf_full_single=use_freq_rf_full_single,
            use_dropout=use_dropout,
        )
        unet_block = UnetSkipConnectionBlock(cfg,ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(cfg, output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

    


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, cfg, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 inject_freq_rf_v0=False, inject_freq_rf_v1a=False,
                 inject_freq_rf_full_single=False, inject_transformer_t8=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            inject_freq_rf_v0 (bool) -- if True, apply FreqRFBlockV0 after down (32×32, inner_nc==ngf*4 for unet_256).
            inject_freq_rf_v1a (bool) -- if True, apply FreqRFBlockV1a (mutually exclusive with other FADC flags).
            inject_freq_rf_full_single (bool) -- if True, apply FreqRFBlockFullSingle.
            inject_transformer_t8 (bool) -- if True, apply TransformerBottleneckT8 after down at 8x8 (512 ch).
        """
        super(UnetSkipConnectionBlock, self).__init__()
        fadc_inject = (inject_freq_rf_v0, inject_freq_rf_v1a, inject_freq_rf_full_single)
        if sum(fadc_inject) > 1:
            raise ValueError("At most one FADC inject_* flag may be True.")
        if inject_transformer_t8 and any(fadc_inject):
            raise ValueError("inject_transformer_t8 cannot be combined with FADC injection.")
        self.outermost = outermost
        self.innermost = innermost
        self.inject_freq_rf_v0 = inject_freq_rf_v0
        self.inject_freq_rf_v1a = inject_freq_rf_v1a
        self.inject_freq_rf_full_single = inject_freq_rf_full_single
        self.inject_transformer_t8 = inject_transformer_t8

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)


        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]

            if cfg.dataset.depth_norm:
                up = [uprelu, upconv, nn.Sigmoid()]  
            else:
                up = [uprelu, upconv, nn.ReLU()]
            model = down + [submodule] + up
            self.model = nn.Sequential(*model)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
            self.model = nn.Sequential(*model)

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if inject_transformer_t8:
                self.down_seq = nn.Sequential(*down)
                self.transformer_bottleneck = TransformerBottleneckT8(
                    d_model=int(getattr(cfg.model, "transformer_channels", 512)),
                    nhead=int(getattr(cfg.model, "transformer_num_heads", 8)),
                    num_layers=int(getattr(cfg.model, "transformer_num_layers", 2)),
                )
                self.submodule = submodule
                self.up_seq = nn.Sequential(*up)
                self.post_dropout = nn.Dropout(0.5) if use_dropout else None
                self.model = None
            elif inject_freq_rf_v1a:
                self.down_seq = nn.Sequential(*down)
                self.freq_rf = FreqRFBlockV1a()
                self.submodule = submodule
                self.up_seq = nn.Sequential(*up)
                self.post_dropout = nn.Dropout(0.5) if use_dropout else None
                self.model = None
            elif inject_freq_rf_full_single:
                self.down_seq = nn.Sequential(*down)
                self.freq_rf = FreqRFBlockFullSingle()
                self.submodule = submodule
                self.up_seq = nn.Sequential(*up)
                self.post_dropout = nn.Dropout(0.5) if use_dropout else None
                self.model = None
            elif inject_freq_rf_v0:
                self.down_seq = nn.Sequential(*down)
                self.freq_rf = FreqRFBlockV0()
                self.submodule = submodule
                self.up_seq = nn.Sequential(*up)
                self.post_dropout = nn.Dropout(0.5) if use_dropout else None
                self.model = None  # forward uses down_seq / freq_rf / submodule / up_seq
            elif use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
                self.model = nn.Sequential(*model)
            else:
                model = down + [submodule] + up
                self.model = nn.Sequential(*model)

    def forward(self, x): 
        if self.outermost:
            return self.model(x)
        if self.inject_transformer_t8:
            xd = self.down_seq(x)
            xd = self.transformer_bottleneck(xd)
            sub = self.submodule(xd)
            out = self.up_seq(sub)
            if self.post_dropout is not None:
                out = self.post_dropout(out)
            return torch.cat([x, out], 1)
        if self.inject_freq_rf_v1a:
            xd = self.down_seq(x)
            xd = self.freq_rf(xd)
            sub = self.submodule(xd)
            out = self.up_seq(sub)
            if self.post_dropout is not None:
                out = self.post_dropout(out)
            return torch.cat([x, out], 1)
        if self.inject_freq_rf_full_single:
            xd = self.down_seq(x)
            xd = self.freq_rf(xd)
            sub = self.submodule(xd)
            out = self.up_seq(sub)
            if self.post_dropout is not None:
                out = self.post_dropout(out)
            return torch.cat([x, out], 1)
        if self.inject_freq_rf_v0:
            xd = self.down_seq(x)
            xd = self.freq_rf(xd)
            sub = self.submodule(xd)
            out = self.up_seq(sub)
            if self.post_dropout is not None:
                out = self.post_dropout(out)
            return torch.cat([x, out], 1)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
