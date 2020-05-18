import torch
import torch.nn as nn
import torch.nn.functional as F

def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_

def functional_conv1(x,name,params):
    weights = params['%s.0.weight'%name]
    bias = params['%s.0.bias'%name]
    x = F.conv2d(x,weights,bias,padding=0)
    x = F.relu(x)
    return x

def functional_conv4(x,name,params,dropout):
    weights = params['%s.0.weight'%name]
    bias = params['%s.0.bias'%name]
    x = F.conv2d(x,weights,bias,padding=1)
    x = F.dropout2d(x,p=dropout)
    x = F.max_pool2d(x,kernel_size=2,stride=2)
    return x

def functional_conv56(x,name,params,dropout):
    weights = params['%s.0.weight'%name]
    bias = params['%s.0.bias'%name]
    x = F.conv2d(x,weights,bias,padding=1)
    x = F.relu(x)
    x = F.dropout2d(x,p=dropout)
    x = F.max_pool2d(x,kernel_size=2,stride=2)
    return x

def functional_conv(x,name,params,pad=1):
    weights = params['%s.weight'%name]
    bias = params['%s.bias'%name]
    x = F.conv2d(x,weights,bias,padding=pad)
    return x

def functional_hconv(x,hname,name,params,pad=1):
    weights = params['%s.%s.weight'%(hname,name)]
    bias = params['%s.%s.bias'%(hname,name)]
    x = F.conv2d(x,weights,bias,padding=pad)
    return x

def functional_hconv_final(x,hname,name,params,pad=1):
    weights = params['%s.%s.0.weight'%(hname,name)]
    bias = params['%s.%s.0.bias'%(hname,name)]
    x = F.conv2d(x,weights,bias,padding=pad)
    x = F.max_pool2d(x,kernel_size=2,stride=2)
    return x

def functional_fc7(x,name,params,dropout):
    weights = params['%s.0.weight'%name]
    bias = params['%s.0.bias'%name]
    x = F.linear(x,weights,bias)
    x = F.relu(x)
    x = F.dropout2d(x,p=dropout)
    return x

class FewShotClassifier(nn.Module):
    def __init__(self,num_class, in_channel=2,
                            length=32,
                            num_joint=10,
                            dropout=0.2):
        super(FewShotClassifier, self).__init__()
        self.num_class = num_class
        self.in_channel = in_channel
        self.length = length
        self.num_joint = num_joint
        self.dropout = dropout
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,64,1,1,padding=0),
            nn.ReLU()
            )
        self.conv2 = nn.Conv2d(64,32,(3,1),1,padding=(1,0))
        self.hconv = HierarchyConv('hconv')
        self.conv4 = nn.Sequential(
            nn.Conv2d(32,64,3,1,padding=1),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2)
        )

        self.convm1 = nn.Sequential(
            nn.Conv2d(in_channel,64,1,1,padding=0),
            nn.ReLU()
            )
        self.convm2 = nn.Conv2d(64,32,(3,1),1,padding=(1,0))
        self.hconvm = HierarchyConv('hconvm')
        self.convm4 = nn.Sequential(
            nn.Conv2d(32,64,3,1,padding=1),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2)
        )
                
        self.conv5 = nn.Sequential(
            nn.Conv2d(128,128,3,1,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128,256,3,1,padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2)
        )

        # scale related to total number of maxpool layer
        scale = 16
        self.fc7 = nn.Sequential(
            nn.Linear(256*(length//scale)*(32//scale),256),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        self.fc8 = nn.Linear(256,self.num_class)

    def forward(self,input):
        output = self.get_feature(input)
        output = self.classify(output)
        return output

    def get_feature(self,input):
        # input: N T J D
        input = input.permute(0,3,1,2)
        N, D, T, J = input.size()
        motion = input[:,:,1::,:]-input[:,:,0:-1,:]
        motion = F.upsample(motion,size=(T,J),mode='bilinear').contiguous()

        out = self.conv1(input)
        out = self.conv2(out)
        out = out.permute(0,3,2,1).contiguous()
        # out: N J T D
        
        # out = self.conv3(out)
        out = self.hconv(out)
        out = self.conv4(out)

        outm = self.convm1(motion)
        outm = self.convm2(outm)
        outm = outm.permute(0,3,2,1).contiguous()
        # outm: N J T D

        # outm = self.convm3(outm)
        outm = self.hconvm(outm)
        outm = self.convm4(outm)

        out = torch.cat((out,outm),dim=1)
        out = self.conv5(out)
        out = self.conv6(out)
        # out:  N J T(T/16) D
        out = out.view(out.size(0),-1)
        return out

    def classify(self,input):
        out = self.fc7(input)
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor
        # N x C (num_class)
        return out

    def functional_forward(self, input, params):
        input = input.permute(0,3,1,2)
        N, D, T, J = input.size()
        motion = input[:,:,1::,:]-input[:,:,0:-1,:]
        motion = F.upsample(motion,size=(T,J),mode='bilinear').contiguous()

        out = functional_conv1(input,'conv1',params)
        out = functional_conv(out,'conv2',params,pad=(1,0))
        out = out.permute(0,3,2,1).contiguous()
        out = self.hconv.functional_forward(out,params)
        out = functional_conv4(out,'conv4',params,self.dropout)

        outm = functional_conv1(motion,'convm1',params)
        outm = functional_conv(outm,'convm2',params,pad=(1,0))
        outm = outm.permute(0,3,2,1).contiguous() 
        outm = self.hconvm.functional_forward(outm,params)
        outm = functional_conv4(outm,'convm4',params,self.dropout)

        out = torch.cat((out,outm),dim=1)
        out = functional_conv56(out,'conv5',params,self.dropout)
        out = functional_conv56(out,'conv6',params,self.dropout)
        out = out.view(out.size(0),-1)

        out = functional_fc7(out,'fc7',params,self.dropout)
        out = F.linear(out,params['fc8.weight'],params['fc8.bias'])

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor
        return out

    def get_optim_policies(self, lr):
        return [
            {'params':self.parameters(),'lr':lr},
        ]

class HierarchyConv(nn.Module):
    def __init__(self,hname):
        super(HierarchyConv,self).__init__()
        self.convla = nn.Conv2d(2,16,3,1,padding=1)
        self.convra = nn.Conv2d(2,16,3,1,padding=1)
        self.conflh = nn.Conv2d(21,16,3,1,padding=1)
        self.confrh = nn.Conv2d(21,16,3,1,padding=1)
        self.convf = nn.Conv2d(70,32,3,1,padding=1)
        self.convl = nn.Conv2d(32,32,3,1,padding=1)
        self.convr = nn.Conv2d(32,32,3,1,padding=1)
        self.parts = 3
        self.conv = nn.Sequential(
            nn.Conv2d(self.parts*32,32,3,1,padding=1),
            nn.MaxPool2d(2)
        )
        self.hname = hname

    def forward(self,input):
        left_arm = input[:,[3,4],:,:]
        right_arm = input[:,[6,7],:,:]
        face = input[:,25:95,:,:]
        left_hand = input[:,95:116,:,:]
        right_hand = input[:,116:137,:,:]
        l1 = self.convla(left_arm) 
        r1 = self.convra(right_arm) 
        l2 = self.conflh(left_hand)
        r2 = self.confrh(right_hand)
        l = torch.cat([l1,l2],1)
        r = torch.cat([r1,r2],1)
        l = self.convl(l)
        r = self.convr(r)
        f = self.convf(face)
        out = torch.cat([l,r,f],1)
        out = self.conv(out)
        return out

    def functional_forward(self,input,params):
        left_arm = input[:,[3,4],:,:]
        right_arm = input[:,[6,7],:,:]
        face = input[:,25:95,:,:]
        left_hand = input[:,95:116,:,:]
        right_hand = input[:,116:137,:,:]
        l1 = functional_hconv(left_arm,self.hname,'convla',params) 
        r1 = functional_hconv(right_arm,self.hname,'convra',params) 
        l2 = functional_hconv(left_hand,self.hname,'conflh',params) 
        r2 = functional_hconv(right_hand,self.hname,'confrh',params)
        l = torch.cat([l1,l2],1)
        r = torch.cat([r1,r2],1)
        l = functional_hconv(l,self.hname,'convl',params)
        r = functional_hconv(r,self.hname,'convr',params)
        f = functional_hconv(face,self.hname,'convf',params)
        out = torch.cat([l,r,f],1)
        out = functional_hconv_final(out,self.hname,'conv',params)
        return out