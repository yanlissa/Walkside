from collections import OrderedDict
import logging
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CONFIG = dict(
    refinement=True,
    multiscale=True,
    dropout=0.05,
    encoder_scale=1.0,
    encoder_padding=0,
)

def stage(modules, channels, block='BASIC'):
    return dict(NUM_MODULES=modules, NUM_BRANCHES=len(channels), NUM_BLOCKS=[4] * len(channels), NUM_CHANNELS=channels, BLOCK=block, FUSE_METHOD='SUM')
cfg = SimpleNamespace(MODEL=SimpleNamespace(ALIGN_CORNERS=False, BNFUNC=nn.BatchNorm2d, HRNET_CHECKPOINT='', OCR_ASPP=False, OCR_EXTRA={'STAGE1': stage(1, [64], 'BOTTLENECK'), 'STAGE2': stage(1, [48, 96]), 'STAGE3': stage(4, [48, 96, 192]), 'STAGE4': stage(3, [48, 96, 192, 384])}))

def Norm2d(channels, **kwargs):
    return nn.BatchNorm2d(channels, **kwargs)

def BNReLU(channels):
    return nn.Sequential(Norm2d(channels), nn.ReLU())

def get_aspp(*args, **kwargs):
    raise RuntimeError('This checkpoint uses OCR without ASPP')


BN_MOMENTUM = 0.1
align_corners = cfg.MODEL.ALIGN_CORNERS
relu_inplace = True

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = Norm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = Norm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = Norm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = Norm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = Norm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out

class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            logger.debug(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            logger.debug(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            logger.debug(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), Norm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), Norm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), Norm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), Norm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM), nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[height_output, width_output], mode='bilinear', align_corners=align_corners)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse
blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

class HighResolutionNet(nn.Module):

    def __init__(self, **kwargs):
        extra = cfg.MODEL.OCR_EXTRA
        super(HighResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = Norm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = Norm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        (self.stage2, pre_stage_channels) = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        (self.stage3, pre_stage_channels) = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        (self.stage4, pre_stage_channels) = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)
        self.high_level_ch = int(np.sum(pre_stage_channels))

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), Norm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM), nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), Norm2d(outchannels, momentum=BN_MOMENTUM), nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), Norm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return (nn.Sequential(*modules), num_inchannels)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        (x0_h, x0_w) = (x[0].size(2), x[0].size(3))
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=align_corners)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=align_corners)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=align_corners)
        feats = torch.cat([x[0], x1, x2, x3], 1)
        return (None, None, feats)


class SpatialGather_Module(nn.Module):

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        (batch_size, c, _, _) = (probs.size(0), probs.size(1), probs.size(2), probs.size(3))
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).unsqueeze(3)
        return ocr_context

class ObjectAttentionBlock(nn.Module):

    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), BNReLU(self.key_channels), nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), BNReLU(self.key_channels))
        self.f_object = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), BNReLU(self.key_channels), nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), BNReLU(self.key_channels))
        self.f_down = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), BNReLU(self.key_channels))
        self.f_up = nn.Sequential(nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0, bias=False), BNReLU(self.in_channels))

    def forward(self, x, proxy):
        (batch_size, h, w) = (x.size(0), x.size(2), x.size(3))
        if self.scale > 1:
            x = self.pool(x)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_channels ** (-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=cfg.MODEL.ALIGN_CORNERS)
        return context

class SpatialOCR_Module(nn.Module):

    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels, key_channels, scale)
        if cfg.MODEL.OCR_ASPP:
            (self.aspp, aspp_out_ch) = get_aspp(in_channels, bottleneck_ch=cfg.MODEL.ASPP_BOT_CH, output_stride=8)
            _in_channels = 2 * in_channels + aspp_out_ch
        else:
            _in_channels = 2 * in_channels
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False), BNReLU(out_channels), nn.Dropout2d(dropout))

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        if cfg.MODEL.OCR_ASPP:
            aspp = self.aspp(feats)
            output = self.conv_bn_dropout(torch.cat([context, aspp, feats], 1))
        else:
            output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


CLASS_NAMES = ("sidewalk", "road", "crosswalk", "background")
CLASS_COLORS = ((61, 61, 245), (10, 153, 0), (255, 4, 4), (0, 0, 0))

class OCRBlock(nn.Module):

    def __init__(self, channels=720):
        super().__init__()
        self.conv3x3_ocr = nn.Sequential(nn.Conv2d(channels, 512, 3, padding=1), BNReLU(512))
        self.ocr_gather_head = SpatialGather_Module(4)
        self.ocr_distri_head = SpatialOCR_Module(512, 256, 512, scale=1, dropout=0.05)
        self.cls_head = nn.Conv2d(512, 4, 1)
        self.aux_head = nn.Sequential(nn.Conv2d(channels, channels, 1), BNReLU(channels), nn.Conv2d(channels, 4, 1))

    def segment(self, features):
        aux = self.aux_head(features)
        features = self.conv3x3_ocr(features)
        context = self.ocr_gather_head(features, aux)
        context = self.ocr_distri_head(features, context)
        return (self.cls_head(context), aux, context)

def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(inplace=True))

class DetailDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.image = conv_block(3, 24)
        self.context = nn.Sequential(nn.Conv2d(512, 24, 1, bias=False), nn.GroupNorm(8, 24), nn.SiLU())
        self.fusion = conv_block(52, 48)
        self.classifier = nn.Conv2d(48, 4, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, images, context, logits):
        context = F.interpolate(self.context(context), size=(256, 256), mode='bilinear', align_corners=False)
        features = torch.cat((self.image(images), context, logits.softmax(1)), 1)
        return logits + self.classifier(self.fusion(features))

class PyramidDetailDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.image_full = conv_block(3, 32)
        self.image_half = conv_block(32, 64)
        self.image_quarter = conv_block(64, 96)
        self.context = nn.Sequential(nn.Conv2d(512, 64, 1, bias=False), nn.GroupNorm(8, 64), nn.SiLU())
        self.high_resolution = nn.Sequential(nn.Conv2d(48, 48, 1, bias=False), nn.GroupNorm(8, 48), nn.SiLU())
        self.fuse_quarter = conv_block(96 + 64 + 48 + 4, 128)
        self.fuse_half = conv_block(128 + 64, 64)
        self.fuse_full = conv_block(64 + 32 + 4, 48)
        self.classifier = nn.Conv2d(48, 4, 1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, images, features, context, logits):
        full = self.image_full(images)
        half = self.image_half(F.avg_pool2d(full, 2))
        quarter = self.image_quarter(F.avg_pool2d(half, 2))
        resize = lambda value, size: F.interpolate(value, size=size, mode='bilinear', align_corners=False)
        probs = logits.softmax(1)
        fused = self.fuse_quarter(torch.cat((quarter, resize(self.context(context), (64, 64)), resize(self.high_resolution(features[:, :48]), (64, 64)), resize(probs, (64, 64))), 1))
        fused = self.fuse_half(torch.cat((resize(fused, (128, 128)), half), 1))
        fused = self.fuse_full(torch.cat((resize(fused, (256, 256)), full, probs), 1))
        return logits + self.classifier(fused)

class TileSegmentationModel(nn.Module):

    def __init__(self, refinement=True, multiscale=False, dropout=0.05, encoder_scale=1.0, encoder_padding=0, pyramid_detail=False):
        super().__init__()
        self.backbone = HighResolutionNet()
        self.ocr = OCRBlock(self.backbone.high_level_ch)
        self.scale_attn = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(512, 256, 3, padding=1, bias=False)), ('bn0', nn.BatchNorm2d(256)), ('re0', nn.ReLU(inplace=True)), ('conv1', nn.Conv2d(256, 256, 3, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(256)), ('re1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(256, 1, 1, bias=False)), ('sig', nn.Sigmoid())]))
        self.dropout = nn.Dropout2d(dropout)
        self.refinement = refinement
        self.multiscale = multiscale
        self.encoder_scale = encoder_scale
        self.encoder_padding = encoder_padding
        if not 0 <= encoder_padding < 256 or encoder_scale <= 0:
            raise ValueError('Expected padding in [0,255] and a positive encoder scale')
        if refinement:
            self.detail = DetailDecoder()
        self.pyramid_detail = pyramid_detail
        if pyramid_detail:
            self.boundary_detail = PyramidDetailDecoder()

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, 256, 256):
            raise ValueError('Expected a batch of RGB tiles [N,3,256,256]')
        pad = self.encoder_padding
        encoder_inputs = F.pad(images, (pad, pad, pad, pad), mode='reflect') if pad else images
        encoder_images = encoder_inputs
        if self.encoder_scale != 1:
            size = round(encoder_inputs.shape[-1] * self.encoder_scale / 32) * 32
            if size < 128:
                raise ValueError('Encoder scale too small')
            encoder_images = F.interpolate(encoder_inputs, size=(size, size), mode='bilinear', align_corners=False)
        features = self.dropout(self.backbone(encoder_images)[2])
        (main, aux, context) = self.ocr.segment(features)
        if self.multiscale:
            low_size = encoder_images.shape[-1] // 2
            low = F.interpolate(encoder_images, size=(low_size, low_size), mode='bilinear', align_corners=False)
            (low_main, low_aux, low_context) = self.ocr.segment(self.dropout(self.backbone(low)[2]))
            attention = self.scale_attn(low_context)
            main = F.interpolate(attention * low_main, size=main.shape[-2:], mode='bilinear', align_corners=False) + (1 - F.interpolate(attention, size=main.shape[-2:], mode='bilinear', align_corners=False)) * main
            aux = F.interpolate(attention * low_aux, size=aux.shape[-2:], mode='bilinear', align_corners=False) + (1 - F.interpolate(attention, size=aux.shape[-2:], mode='bilinear', align_corners=False)) * aux
        main = F.interpolate(main, size=encoder_inputs.shape[-2:], mode='bilinear', align_corners=False)
        aux = F.interpolate(aux, size=encoder_inputs.shape[-2:], mode='bilinear', align_corners=False)
        if pad:
            (main, aux) = (main[:, :, pad:pad + 256, pad:pad + 256], aux[:, :, pad:pad + 256, pad:pad + 256])
            crop = round(context.shape[-1] * pad / encoder_inputs.shape[-1])
            if crop:
                context = context[:, :, crop:-crop, crop:-crop]
                features = features[:, :, crop:-crop, crop:-crop]
        if self.refinement:
            main = self.detail(images, context, main)
        if self.pyramid_detail:
            main = self.boundary_detail(images, features, context, main)
        return {'main': main, 'aux': aux}