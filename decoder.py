import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.functional import interpolate


def conv(in_f, out_f, kernel_size, stride=1, pad='zero'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

    layers = filter(lambda x: x is not None, [padder, convolver]) #extract values of non-None padder and convolver 
    return nn.Sequential(*layers)

     
        
class Downsample(torch.nn.Module): 
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Downsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)


class autoencodernet(torch.nn.Module):
    def __init__(self, num_output_channels, num_channels_up, need_sigmoid=True, pad='reflection', upsample_mode='bilinear', bn_affine=True, decodetype='upsample', kernel_size=1):
        super(autoencodernet,self).__init__()
        self.decodetype = decodetype
        
        n_scales = len(num_channels_up)
        # print('n_scales=', n_scales, 'num_channels_up=', num_channels_up)
        
        if decodetype=='upsample':
            #decoder
            self.decoder = nn.Sequential()

            for i in range(n_scales-1):
                
                #if i!=0:
                module_name = 'dconv'+str(i)    
                self.decoder.add_module(module_name, conv(num_channels_up[i], num_channels_up[i+1], kernel_size, 1, pad=pad))

                if i != len(num_channels_up)-1:        
                    module_name = 'drelu' + str(i)
                    self.decoder.add_module(module_name,nn.ReLU())   
                    module_name = 'dbn' + str(i)
                    self.decoder.add_module(module_name,nn.BatchNorm2d(num_channels_up[i+1], affine=bn_affine))        

                module_name = 'dups' + str(i)
                self.decoder.add_module(module_name,nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True))

            module_name = 'dconv' + str(i+1)
            self.decoder.add_module(module_name, conv(num_channels_up[-1], num_output_channels, kernel_size, 1, pad=pad))
        
            if need_sigmoid:
                self.decoder.add_module('sig',nn.Sigmoid())
                
        #encoder
        self.encoder = nn.Sequential()
        module_name = 'uconv'+str(n_scales-1)   
        self.encoder.add_module(module_name,conv(64,num_channels_up[-1], 1, pad=pad))
        
        for i in range(n_scales-2,-1,-1):
            
            if i != len(num_channels_up)-1:  
                module_name = 'urelu' + str(i)
                self.encoder.add_module(module_name,nn.ReLU())
                module_name = 'ubn' + str(i)
                self.encoder.add_module(module_name,nn.BatchNorm2d(num_channels_up[i+1], affine=bn_affine))     
                
            module_name = 'uconv'+str(i)
            self.encoder.add_module(module_name,conv(num_channels_up[i+1], num_channels_up[i],  1, 1, pad=pad))    
            module_name = 'udns'+str(i)
            self.encoder.add_module(module_name,Downsample(scale_factor=0.5, mode=upsample_mode, align_corners=True))

        if decodetype=='transposeconv':
            #convolutional decoder
            self.convdecoder = nn.Sequential()
            
            for i in range(n_scales-1):
                module_name = 'cdconv'+str(i) 
                
                if i==0:
                    self.convdecoder.add_module(module_name,conv(num_channels_up[i], num_channels_up[i+1], 1, 1, pad=pad))
                else:
                    self.convdecoder.add_module(module_name,nn.ConvTranspose2d(num_channels_up[i], num_channels_up[i+1],2,2)) 

                if i != len(num_channels_up)-1:        
                    module_name = 'cdrelu' + str(i)
                    self.convdecoder.add_module(module_name,nn.ReLU())   
                    module_name = 'cdbn' + str(i)
                    self.convdecoder.add_module(module_name,nn.BatchNorm2d(num_channels_up[i+1], affine=bn_affine))        

            module_name = 'cdconv' + str(i+2)
            self.convdecoder.add_module(module_name,nn.ConvTranspose2d(num_channels_up[-1], num_output_channels, 2, 2)) 
            
            if need_sigmoid:
                self.convdecoder.add_module('sig',nn.Sigmoid())
    def forward(self,x):
        if self.decodetype=='upsample':
            x = self.decoder(x)
        elif self.decodetype=='transposeconv':
            x = self.convdecoder(x)
        return x


class BasicBlock(torch.nn.Module):
    def __init__(self, stageloss_w, patch_size, channel_list):
        super(BasicBlock, self).__init__()

        self.patch_size = patch_size
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.stageloss_w = stageloss_w
        self.stage_coeff = nn.Parameter(torch.Tensor([1.0]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(channel_list[0], 1, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(channel_list[1], channel_list[0], 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(channel_list[2], channel_list[1], 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, channel_list[2], 3, 3)))

    def forward(self, x):
        x_input = x.view(-1, 1, self.patch_size, self.patch_size) # (batch, channel, H, W)

        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr)) # soft threshold

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_pred = x_backward.view(-1, self.patch_size**2)
        if self.stageloss_w:
            x_pred = self.stage_coeff * x_pred

        # inverse the function to calculate the symmetric loss
        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return x_pred, symloss

class ISTANet(torch.nn.Module):
    def __init__(self, num_layer, stageloss_w, patch_size, channel_list):
        super(ISTANet, self).__init__()
        onelayer = []
        self.num_layer = num_layer
        for i in range(num_layer):
            onelayer.append(BasicBlock(stageloss_w, patch_size, channel_list))
        self.fcs = nn.ModuleList(onelayer)
        # self.fcs = nn.Sequential(*onelayer)

    def forward(self, x):
        layers_sym = [] # for computing symmetric loss
        x_outputs = []
        for i in range(self.num_layer):
            x, layer_sym = self.fcs[i](x)
            x_outputs.append(x)
            layers_sym.append(layer_sym)
        x_final = x
        return x_final, layers_sym, x_outputs

class BNNet(nn.Module):
    def __init__(self,num_channel):
        super(BNNet, self).__init__()
        self.bn = nn.BatchNorm2d(num_channel)

    def forward(self, input_data):
        output_data = self.bn(input_data)
        return output_data