from __future__ import print_function

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum


class ConvReg(nn.Module):
    """Convolutional regression for FitNet (feature-map layer)"""
    def __init__(self, s_shape, t_shape):
        super(ConvReg, self).__init__()
        _, s_C, s_H, s_W = s_shape
        _, t_C, t_H, t_W = t_shape
        self.s_H = s_H
        self.t_H = t_H
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        x = self.conv(x)
        if self.s_H == 2 * self.t_H or self.s_H * 2 == self.t_H or self.s_H >= self.t_H:
            return self.relu(self.bn(x)), t
        else:
            return self.relu(self.bn(x)), F.adaptive_avg_pool2d(t, (self.s_H, self.s_H))
            
class SelfA(nn.Module):
    """Cross-layer Self Attention"""
    def __init__(self, feat_dim, s_n, t_n, soft, factor=4): 
        super(SelfA, self).__init__()

        self.soft = soft
        self.s_len = len(s_n)
        self.t_len = len(t_n)
        self.feat_dim = feat_dim

        # query and key mapping
        for i in range(self.s_len):
            setattr(self, 'query_'+str(i), MLPEmbed(feat_dim, feat_dim//factor))
        for i in range(self.t_len):
            setattr(self, 'key_'+str(i), MLPEmbed(feat_dim, feat_dim//factor))
        
        for i in range(self.s_len):
            for j in range(self.t_len):
                setattr(self, 'regressor'+str(i)+str(j), Proj(s_n[i], t_n[j]))
               
    def forward(self, feat_s, feat_t):
        
        sim_s = list(range(self.s_len))
        sim_t = list(range(self.t_len))
        bsz = self.feat_dim

        # similarity matrix
        for i in range(self.s_len):
            sim_temp = feat_s[i].reshape(bsz, -1)
            sim_s[i] = torch.matmul(sim_temp, sim_temp.t())
        for i in range(self.t_len):
            sim_temp = feat_t[i].reshape(bsz, -1)
            sim_t[i] = torch.matmul(sim_temp, sim_temp.t())
        
        # calculate student query
        proj_query = self.query_0(sim_s[0])
        proj_query = proj_query[:, None, :]
        for i in range(1, self.s_len):
            temp_proj_query = getattr(self, 'query_'+str(i))(sim_s[i])
            proj_query = torch.cat([proj_query, temp_proj_query[:, None, :]], 1)
        
        # calculate teacher key
        proj_key = self.key_0(sim_t[0])
        proj_key = proj_key[:, :, None]
        for i in range(1, self.t_len):
            temp_proj_key = getattr(self, 'key_'+str(i))(sim_t[i])
            proj_key =  torch.cat([proj_key, temp_proj_key[:, :, None]], 2)
        
        # attention weight: batch_size X No. stu feature X No.tea feature
        energy = torch.bmm(proj_query, proj_key)/self.soft
        attention = F.softmax(energy, dim = -1)
        
        # feature dimension alignment
        proj_value_stu = []
        value_tea = []
        for i in range(self.s_len):
            proj_value_stu.append([])
            value_tea.append([])
            for j in range(self.t_len):            
                s_H, t_H = feat_s[i].shape[2], feat_t[j].shape[2]
                if s_H > t_H:
                    source = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
                    target = feat_t[j]
                elif s_H <= t_H:
                    source = feat_s[i]
                    target = F.adaptive_avg_pool2d(feat_t[j], (s_H, s_H))
                
                proj_value_stu[i].append(getattr(self, 'regressor'+str(i)+str(j))(source))
                value_tea[i].append(target)

        return proj_value_stu, value_tea, attention

class Proj(nn.Module):
    """feature dimension alignment by 1x1, 3x3, 1x1 convolutions"""
    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(Proj, self).__init__()
        self.num_mid_channel = 2 * num_target_channels
        
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x

class MLPEmbed(nn.Module):
    """non-linear mapping for attention calculation"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)
        self.regressor = nn.Sequential(
            nn.Linear(dim_in, 2 * dim_out),
            self.l2norm,
            nn.ReLU(inplace=True),
            nn.Linear(2 * dim_out, dim_out),
            self.l2norm,
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        
        return x

class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class SRRL(nn.Module):
    """ICLR-2021: Knowledge Distillation via Softmax Regression Representation Learning"""
    def __init__(self, *, s_n, t_n): 
        super(SRRL, self).__init__()
                
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        
        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        ))
        
    def forward(self, feat_s, cls_t):
        
        feat_s = feat_s.unsqueeze(-1).unsqueeze(-1)
        temp_feat = self.transfer(feat_s)
        trans_feat_s = temp_feat.view(temp_feat.size(0), -1)

        pred_feat_s=cls_t(trans_feat_s)

        return trans_feat_s, pred_feat_s

class MHSA(nn.Module):
    def __init__(self, n_dims, width=8, height=8, heads=8, pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                                             requires_grad=True)
            self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                                             requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(
                0, 1, 3, 2)

            content_position = torch.matmul(content_position, q)
            content_position = content_position if (
                    content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
            assert (content_content.shape == content_position.shape)
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        return out

class StripPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))#1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))#H*1
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                     nn.BatchNorm2d(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                     nn.BatchNorm2d(inter_channels),
                                     nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels))
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x) [64,256,8,8]
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs) #结构图的1*W的部分
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs) #结构图的H*1的部分
        x4 = self.conv4(F.relu_(x2 + x3))#结合1*W和H*1的特征
        out = self.conv5(x4)
        return F.relu_(x + out)#将输出的特征与原始输入特征结合


# 构建SPP层(空间金字塔池化层)
class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='avg_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (
            math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=2, padding=1, bias=True, activation='relu', norm=True):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
    def forward(self, x):
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=2, padding=1, bias=True, activation='relu', norm=True):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


class SimKD(nn.Module):
    """CVPR-2022: Knowledge Distillation with the Reused Teacher Classifier"""
    def __init__(self, *, s_n, t_n, factor=2, ):
        super(SimKD, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False, groups=1)

        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False,
                             groups=groups)

        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n // factor),
            nn.BatchNorm2d(t_n // factor),
            nn.ReLU(inplace=True),

            conv3x3(t_n // factor, t_n // factor),
            nn.BatchNorm2d(t_n // factor),
            nn.ReLU(inplace=True),

            conv1x1(t_n // factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        ))

    def forward(self, feat_s, feat_t, cls_t):
        s_H, t_H = feat_s.shape[2], feat_t.shape[2]
        if s_H > t_H:
            source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
            target = feat_t
        else:
            source = feat_s
            target = F.adaptive_avg_pool2d(feat_t, (s_H, s_H))
        trans_feat_t = target

        # Channel Alignment
        trans_feat_s = getattr(self, 'transfer')(source)

        # Prediction via Teacher Classifier
        temp_feat = self.avg_pool(trans_feat_s)
        temp_feat = temp_feat.view(temp_feat.size(0), -1)
        pred_feat_s = cls_t(temp_feat)

        return trans_feat_s, trans_feat_t, pred_feat_s



class mfakd(nn.Module):
    def __init__(self, *, s_n, t_n, factor=2,):
        super(mfakd, self).__init__()
       
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU(inplace=True)


        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False,groups=1)
        def conv3x3(in_channels, out_channels, stride=1,groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)
        # sandglassblock
        setattr(self, 'transfer1', nn.Sequential(
            conv3x3(s_n,t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),

            conv1x1(t_n,t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),

            conv1x1(t_n//factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),

            conv3x3(t_n,t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
            ))
        # sandglassblock+
        setattr(self, 'transfer2', nn.Sequential(
            conv3x3(s_n,t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),

            conv1x1(t_n,t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),

            conv1x1(t_n//factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),

            conv3x3(t_n,t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),

            conv1x1(t_n, t_n // factor),
            nn.BatchNorm2d(t_n // factor),
            nn.ReLU(inplace=True),

            conv1x1(t_n // factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),

            conv3x3(t_n,t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
            ))
        # bottleneck
        setattr(self, 'transfer3', nn.Sequential(
            conv1x1(s_n, t_n // factor),
            nn.BatchNorm2d(t_n // factor),
            nn.ReLU(inplace=True),

            conv3x3(t_n // factor,t_n // factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),

            conv1x1(t_n//factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
            ))

    def forward(self, feat_s, feat_t, cls_t):

        # Spatial Dimension Alignment
        s_H, t_H = feat_s.shape[2], feat_t.shape[2]
        if s_H > t_H:
            source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
            target = feat_t
        elif s_H == t_H:
            source = feat_s
            target = feat_t
        # Spatial dimention Alignment
        else:
            source = F.interpolate(feat_s,(t_H,t_H),mode='nearest')
            target = feat_t
        trans_feat_t = target

        # Channel Alignment
        trans_feat_s1 = getattr(self, 'transfer1')(source)
        trans_feat_s2 = getattr(self, 'transfer2')(source)
        trans_feat_s3 = getattr(self, 'transfer3')(source)
        trans_feat_s = (trans_feat_s1+trans_feat_s2+trans_feat_s3)/3.0

        # Prediction via Teacher Classifier
        temp_feat = self.avg_pool(trans_feat_s)
        temp_feat = temp_feat.view(temp_feat.size(0), -1)
        pred_feat_s = cls_t(temp_feat)

        return trans_feat_s, trans_feat_t, pred_feat_s
