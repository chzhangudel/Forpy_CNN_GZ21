#!/bin/env python

from abc import abstractmethod,ABC
import torch
from torch.nn import  Module, Parameter
from torch import nn
import numpy as np
import math
from torch.nn.functional import softplus

# GPU setup
args_no_cuda = False #True when manually turn off cuda
use_cuda = not args_no_cuda and torch.cuda.is_available()
if use_cuda:
    print('device for inference on',torch.cuda.device_count(),'GPU(s)')
else:
    print('device for inference on CPU')


cem_testing_flag = False
if cem_testing_flag:
    import os
    from constants.paths import ONLINE_MODELS
    fn = f'cem_20230418.pth'
    path = os.path.join(ONLINE_MODELS,fn)
    nn_load_main_file = path
else:
    nn_load_main_file= '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/cem_20230418.pth'

nn_load_name = 'gaussian_four_regions'

u_scale = 1/0.1 #0.09439346225350978
v_scale = 1/0.1 #0.07252696573672539
Su_scale = 1e-7 #4.9041400042653195e-08
Sv_scale = 1e-7 #4.8550991806254025e-08

    


class Transform(Module, ABC):
    """Abstract Base Class for all transforms"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def transform(self, input):
        pass

    def forward(self, input_):
        return self.transform(input_)

    @abstractmethod
    def __repr__(self):
        pass
    
    def __call__(self,input_):
        return self.forward(input_)

class PrecisionTransform(Transform):
    def __init__(self, min_value=0.1):
        super().__init__()
        self.min_value = Parameter(torch.tensor(min_value))
        self.indices = slice(2,4)

    @property
    def min_value(self):
        return softplus(self._min_value)

    @min_value.setter
    def min_value(self, value):
        self._min_value = Parameter(torch.tensor(value))

    @property
    def indices(self):
        """Return the indices transformed"""
        return self._indices

    @indices.setter
    def indices(self, values):
        self._indices = values

    def transform(self, input_):
        # Split in sections of size 2 along channel dimension
        # Careful: the split argument is the size of the sections, not the
        # number of them (although does not matter for 4 channels)
        result = torch.clone(input_)
        result[:, self.indices, :, :] = self.transform_precision(
            input_[:, self.indices, :, :]) + self.min_value
        return result

    @staticmethod
    @abstractmethod
    def transform_precision(precision):
        pass

class SquareTransform(PrecisionTransform):
    def __init__(self, min_value=0.1):
        super().__init__(min_value)

    @staticmethod
    def transform_precision(precision):
        return precision**2

    def __repr__(self):
        return ''.join(('SquareTransform(', str(self.min_value), ')'))
    
class SoftPlusTransform(PrecisionTransform):
    def __init__(self, min_value=0.1):
        super().__init__(min_value)

    @staticmethod
    def transform_precision(precision):
        return softplus(precision)

    def __repr__(self):
        return ''.join(('SoftPlusTransform(', str(self.min_value), ')'))

class Layer:
    def __init__(self,nn_layers:list) -> None:
        self.nn_layers =nn_layers
        self.section = []
    def add(self,nn_obj):
        self.section.append(len(self.nn_layers))
        self.nn_layers.append(nn_obj)
    def __call__(self,x):
        for j in self.section:
            x = self.nn_layers[j](x)
        return x

class CNN_Layer(Layer):
    def __init__(self,nn_layers:list,widthin,widthout,kernel,batchnorm,nnlnr) -> None:
        super().__init__(nn_layers)
        self.add(nn.Conv2d(widthin,widthout,kernel))
        if batchnorm:
            self.add(nn.BatchNorm2d(widthout))
        if nnlnr:
            self.add(nn.ReLU(inplace = True))
class SoftPlus_Layer(Layer):
    def __init__(self,nn_layers:list,split,min_value = 0) -> None:
        super().__init__(nn_layers)
        self.add(nn.Softplus())
        self.min_value = min_value
        self.split = split
    def __call__(self, x):
        if self.split>1:
            xs = list(torch.split(x,x.shape[1]//self.split,dim=1))
            p = super().__call__(xs[-1])
            p = p + self.min_value
            xs[-1] = p
            return tuple(xs)
        return super().__call__(x)

class Square_Layer(Layer):
    def __init__(self,nn_layers:list,split,min_value = 0) -> None:
        super().__init__(nn_layers)
        st = SquareTransform(min_value = min_value)
        self.add(st)
        self.min_value = min_value
        self.split = split
    def __call__(self, x):
        if self.split>1:
            x = super().__call__(x)
            xs = list(torch.split(x,x.shape[1]//self.split,dim=1))            
            return tuple(xs)
        return super().__call__(x)
class SoftPlus_Layer_With_Constant(Layer):
    def __init__(self,nn_layers:list,split,min_value = 0) -> None:
        super().__init__(nn_layers)
        st = SoftPlusTransform(min_value = min_value)
        self.add(st)
        self.min_value = min_value
        self.split = split
    def __call__(self, x):
        if self.split>1:
            x = super().__call__(x)
            xs = list(torch.split(x,x.shape[1]//self.split,dim=1))            
            return tuple(xs)
        return super().__call__(x)
    
class Sequential(Layer):
    def __init__(self,nn_layers,widths,kernels,batchnorm,final_activation ,min_precision,split = 1):
        super().__init__(nn_layers)
        self.sections = []
        spread = 0
        self.nlayers = len(kernels)
        for i in range(self.nlayers):
            spread+=kernels[i]-1
        self.spread = spread//2
        for i in range(self.nlayers):
            self.sections.append(CNN_Layer(nn_layers,widths[i],widths[i+1],kernels[i],batchnorm[i], i < self.nlayers - 1))
        if final_activation == 'softplus':
            self.sections.append(SoftPlus_Layer(nn_layers,split,min_value = min_precision))
        elif final_activation == 'square':
            self.sections.append(Square_Layer(nn_layers,split,min_value = min_precision))
        elif final_activation == 'softplus_with_constant':
            self.sections.append(SoftPlus_Layer_With_Constant(nn_layers,split,min_value = min_precision))
    def __call__(self, x):
        for lyr in self.sections:
            x = lyr.__call__(x)
        return x

class CNN(nn.Module):
    def __init__(self,widths = None,kernels = None,batchnorm = None,seed = None,final_activation = None,min_precision = 0 ,**kwargs):
        super(CNN, self).__init__()
        torch.manual_seed(seed)
        self.nn_layers = nn.ModuleList()
        
        self.sequence = \
            Sequential(self.nn_layers, widths,kernels,batchnorm,final_activation,min_precision ,split = 2)
        spread = 0
        for k in kernels:
            spread += (k-1)/2
        self.spread = int(spread)
    def forward(self,x1):
        return self.sequence(x1)



statedict = torch.load(nn_load_main_file)
modeldict,statedict = statedict[nn_load_name]
cnn=CNN(**modeldict)
cnn.load_state_dict(statedict)
cnn.eval()



def MOM6_testNN(uv,pe,pe_num,index):
    global cnn,gpu_id,u_scale,v_scale,Su_scale,Sv_scale
    # start_time = time.time()
    # print('PE number is',pe_num)
    # print('PE is',pe)
    # print(u.shape,v.shape)
    #normalize the input by training scaling
    u= uv[0,:,:,:]*u_scale
    v= uv[1,:,:,:]*v_scale
    x = np.array([np.squeeze(u),np.squeeze(v)])
    if x.ndim==3:
        x = x[:,:,:,np.newaxis]
    x = x.astype(np.float32)
    # print(x.shape)
    x = x.transpose((3,0,1,2)) # new the shape is (nk,2,ni,nj)
    x = torch.from_numpy(x) # quite faster than x = torch.tensor(x)
    if use_cuda:
        if not next(cnn.parameters()).is_cuda:
            gpu_id = int(pe/math.ceil(pe_num/torch.cuda.device_count()))
            print('GPU id is:',gpu_id)
            cnn = cnn.cuda(gpu_id)
        x = x.cuda(gpu_id)
    with torch.no_grad():
        # start_time = time.time()
        outs = cnn.forward(x)
        if isinstance(outs,tuple):
            mean,precision = outs
        else:
            mean,precision = torch.split(outs,2,dim = 1)
        # end_time = time.time()
    if use_cuda:
        mean = mean.to('cpu')
        precision = precision.to('cpu')
    mean = mean.numpy().astype(np.float64)
    std = np.sqrt(1/precision.numpy().astype(np.float64))
    # At this point, python out shape is (nk,4,ni,nj)
    # Comment-out is tranferring arraies into F order
    # convert out to (ni,nj,nk)
    mean = mean.transpose((1,2,3,0)) # new the shape is (4,ni,nj,nk)
    std = std.transpose((1,2,3,0))
    dim = np.shape(mean)
    Sxy = np.zeros((6,dim[1],dim[2],dim[3])) # the shape is (2,ni,nj,nk)
    epsilon_x = np.random.normal(0, 1, size=(dim[1],dim[2]))
    epsilon_x = np.dstack([epsilon_x]*dim[3])
    epsilon_y = np.random.normal(0, 1, size=(dim[1],dim[2]))
    epsilon_y = np.dstack([epsilon_y]*dim[3])				  
    Sxy[0,:,:,:] = (mean[0,:,:,:] )*Su_scale
    Sxy[1,:,:,:] = (mean[1,:,:,:] )*Sv_scale
    Sxy[2,:,:,:] = mean[0,:,:,:]*Su_scale
    Sxy[3,:,:,:] = mean[1,:,:,:]*Sv_scale
    #    Sxy[4,:,:,:] = std[0,:,:,:]*Su_scale
    #    Sxy[5,:,:,:] = std[1,:,:,:]*Sv_scale
    """
    np.savetxt('Sx_mean_cem.txt',Sxy[2,:,:,0])
    np.savetxt('Sy_mean_cem.txt',Sxy[3,:,:,0])
    np.savetxt('Sx_std_cem.txt',Sxy[4,:,:,0])
    np.savetxt('Sy_std_cem.txt',Sxy[5,:,:,0])
    np.savetxt('WH_u_cem.txt',uv[0,:,:,0])
    np.savetxt('WH_v_cem.txt',uv[1,:,:,0])
    """
    return Sxy 