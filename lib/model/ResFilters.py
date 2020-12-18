import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from ..net_util import *

class ResNetMidFeat(nn.Module):
    

    def __init__(self, depth, pretrained):
        super(ResNetMidFeat, self).__init__()

        __resnet_dict = {
            'resnet18': torchvision.models.resnet18,
            'resnet34': torchvision.models.resnet34,
            'resnet50': torchvision.models.resnet50,
            'resnet101': torchvision.models.resnet101,
            'resnet152': torchvision.models.resnet152,
        }
        
        module_list = list(__resnet_dict['resnet%d'%depth](pretrained=pretrained).children())
        self.ds_block1 = nn.Sequential(*(module_list[0:3]))
        self.ds_block2 = module_list[3]
        self.resblock1 = module_list[4]
        self.resblock2 = module_list[5]
        self.resblock3 = module_list[6]
        self.resblock4 = module_list[7]
        self.avg_pool = module_list[8]

        feat_channels = 256
        block1_outc = self.resblock2[0].conv1.in_channels
        block2_outc = block1_outc * 2
        block3_outc = block2_outc * 2
        block4_outc = block3_outc * 2

        self.reduct1x1_1 = nn.Conv2d(block1_outc, feat_channels, kernel_size=1)
        self.reduct1x1_2 = nn.Conv2d(block2_outc, feat_channels, kernel_size=1)
        self.reduct1x1_3 = nn.Conv2d(block3_outc, feat_channels, kernel_size=1)
        self.reduct1x1_4 = nn.Conv2d(block4_outc, feat_channels, kernel_size=1)
        
        #print(self.resblock1)
        #print('[HERE: In lib/model/ResFilters/ResNetMiddlePart] resblock.in_channels = %d'%self.resblock1[0].conv1.in_channels)

    def forward(self, x):
        feat_list = []
        tmpx = None
        normx = None

        tmpx = self.ds_block1(x)
        normx = self.ds_block2(tmpx)

        x = normx
        
        x = self.resblock1(x)
        y = self.reduct1x1_1(x)
        feat_list.append(y)

        x = self.resblock2(x)
        y = self.reduct1x1_2(x)
        feat_list.append(y)

        x = self.resblock3(x)
        y = self.reduct1x1_3(x)
        feat_list.append(y)

        x = self.resblock4(x)
        y = self.reduct1x1_4(x)
        feat_list.append(y)

        return feat_list, tmpx, normx

class ResFilter(nn.Module):
    def __init__(self, opt):
        super(ResFilter, self).__init__()
        #self.num_modules = opt.num_stack

        self.opt = opt

        # Base part
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        #if self.opt.norm == 'batch':
        #    self.bn1 = nn.BatchNorm2d(64)
        #elif self.opt.norm == 'group':
        #    self.bn1 = nn.GroupNorm(32, 64)

        #if self.opt.hg_down == 'conv64':
        #    self.conv2 = ConvBlock(64, 64, self.opt.norm)
        #    self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        #elif self.opt.hg_down == 'conv128':
        #    self.conv2 = ConvBlock(64, 128, self.opt.norm)
        #    self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        #elif self.opt.hg_down == 'ave_pool':
        #    self.conv2 = ConvBlock(64, 128, self.opt.norm)
        #else:
        #    raise NameError('Unknown Fan Filter setting!')

        #self.conv3 = ConvBlock(128, 128, self.opt.norm)
        #self.conv4 = ConvBlock(128, 256, self.opt.norm)

        # resnet part
        self.resnet = ResNetMidFeat(depth=34, pretrained=False)

    def forward(self, x):
        """
        if self.opt.debug:
            print('[HERE: In lib/model/ResFilters] -----entering a forward pass of ResFilters-----')
            print('[HERE: In lib/model/ResFilters] input.shape:', x.shape)

        x = F.relu(self.bn1(self.conv1(x)), True)
        tmpx = x
        if self.opt.debug:
            print('[HERE: In lib/model/ResFilters] tempx.shape:', tmpx.shape)
        
        if self.opt.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.opt.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')

        normx = x
        if self.opt.debug:
            print('[HERE: In lib/model/ResFilters] normx.shape:', normx.shape)

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        if self.opt.debug:
            print('[HERE: In lib/model/ResFilters] previous.shape:', previous.shape)

        outputs = self.resnet_mid(previous)

        if self.opt.debug:    
            print('[HERE: In lib/model/ResFilters] len(outputs):', len(outputs))
            for i, out in enumerate(outputs):
                print('[HERE: In lib/model/ResFilters] outputs[%d].shape:'%i, outputs[i].shape)
        
            print('[HERE: In lib/model/ResFilters] -----exiting a forward pass of ResFilters-----')

        return outputs, tmpx.detach(), normx
        """
        return self.resnet(x)
