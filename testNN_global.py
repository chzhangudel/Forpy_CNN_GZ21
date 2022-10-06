#!/bin/env python

import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import math
import time

# GPU setup
args_no_cuda = False #True when manually turn off cuda
use_cuda = not args_no_cuda and torch.cuda.is_available()
if use_cuda:
    print('device for inference on',torch.cuda.device_count(),'GPU(s)')
else:
    print('device for inference on CPU')

nn_load_file='/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/best-model'
filters=[5, 5, 3, 3, 3, 3, 3, 3]
widths=[128, 64, 32, 32, 32, 32, 32, 4]

u_scale=1/0.10278768092393875
v_scale=1/0.07726840674877167
world_radius_in_meters=6.371e6
angle_to_meters=world_radius_in_meters*2*np.pi/360
Su_scale=0.004745704121887684/angle_to_meters
Sv_scale=0.004386111628264189/angle_to_meters
# print('suscale',Su_scale)
# print('svscale',Sv_scale)

#load the neural network
class CNN(nn.Module):
    def __init__(self,filter_size=[5, 5, 3, 3, 3, 3, 3, 3],\
                     width=[128, 64, 32, 32, 32, 32, 32, 4],\
                        inchan=2,cuda_flag=False):
        super(CNN, self).__init__()
        self.nn_layers = nn.ModuleList()
        self.filter_size=filter_size
        self.num_layers=len(filter_size)
        
        if cuda_flag:
            device = "cuda:0" 
        else:  
            device = "cpu"  
        
        self.nn_layers.append(nn.Conv2d(inchan, width[0], filter_size[0]).to(device) )
        for i in range(1,self.num_layers):
            self.nn_layers.append(nn.BatchNorm2d(width[i-1]).to(device) )
            self.nn_layers.append(nn.Conv2d(width[i-1], width[i], filter_size[i]).to(device) )
        self.nn_layers.append(nn.Softplus().to(device))
    def forward(self, x):
        cn=0
        while cn<len(self.nn_layers)-2:
            x = self.nn_layers[cn](x)
            cn+=1
            x = F.relu(self.nn_layers[cn](x))
            cn+=1
        x=self.nn_layers[cn](x)
        mean,precision=torch.split(x,x.shape[1]//2,dim=1)
        precision=self.nn_layers[-1](precision)
        out=torch.cat([mean,precision],dim=1)
        return out

nn=CNN(cuda_flag=False)
nn.load_state_dict(torch.load(nn_load_file,map_location='cpu'))
nn.eval()

def MOM6_testNN(u,v,pe,pe_num):
   global nn,gpu_id,u_scale,v_scale,Su_scale,Sv_scale
   # start_time = time.time()
   # print('PE number is',pe_num)
   # print('PE is',pe)
   # print(u.shape,v.shape)
   #normalize the input by training scaling
   u=u*u_scale
   v=v*v_scale
   x = np.array([np.squeeze(u),np.squeeze(v)])
   if x.ndim==3:
     x = x[:,:,:,np.newaxis]
   x = x.astype(np.float32)
   x = x.transpose((3,0,1,2)) # new the shape is (nk,2,ni,nj)
   x = torch.from_numpy(x) # quite faster than x = torch.tensor(x)
   if use_cuda:
       if not next(nn.parameters()).is_cuda:
          gpu_id = int(pe/math.ceil(pe_num/torch.cuda.device_count()))
          print('GPU id is:',gpu_id)
          nn = nn.cuda(gpu_id)
       x = x.cuda(gpu_id)
   with torch.no_grad():
       # start_time = time.time()
       out = nn.forward(x)
       # end_time = time.time()
   if use_cuda:
       out = out.to('cpu')
   out = out.numpy().astype(np.float64)
   # At this point, python out shape is (nk,4,ni,nj)
   # Comment-out is tranferring arraies into F order
   """
   print(out.shape)
   dim = np.shape(out)
   out = out.flatten(order='F')
   out = out.reshape(dim[0],dim[1],dim[2],dim[3], order='F')
   """
   # convert out to (ni,nj,nk)
   out = out.transpose((1,2,3,0)) # new the shape is (4,ni,nj,nk)
   dim = np.shape(out)
   # print(dim)
   Sxy = np.zeros((2,dim[1],dim[2],dim[3])) # the shape is (2,ni,nj,nk)
   epsilon_x = np.random.normal(0, 1, size=(dim[1],dim[2]))
   epsilon_x = np.dstack([epsilon_x]*dim[3])
   epsilon_y = np.random.normal(0, 1, size=(dim[1],dim[2]))
   epsilon_y = np.dstack([epsilon_y]*dim[3])
   # if pe==0:
   #   print(scaling)
   """
   # mean output
   Sxy[0,:,:,:] = (out[0,:,:,:])*Su_scale
   Sxy[1,:,:,:] = (out[1,:,:,:])*Sv_scale
   # std output
   Sxy[0,:,:,:] = (epsilon_x/out[2,:,:,:])*Su_scale
   Sxy[1,:,:,:] = (epsilon_y/out[3,:,:,:])*Sv_scale
   """
   # full output
   Sxy[0,:,:,:] = (out[0,:,:,:] + epsilon_x*np.sqrt(1/out[2,:,:,:]))*Su_scale
   Sxy[1,:,:,:] = (out[1,:,:,:] + epsilon_y*np.sqrt(1/out[3,:,:,:]))*Sv_scale
   """
   # scaling the parameters for upper and lower layers
   Sxy[:,:,:,0]=Sxy[:,:,:,0]*0.8
   Sxy[:,:,:,1]=Sxy[:,:,:,1]*1.5
   """
   """
   np.savetxt('Sx_mean.txt',out[0,:,:,0])
   np.savetxt('Sx_std.txt',out[2,:,:,0])
   np.savetxt('WH_u.txt',u[:,:,1])
   np.savetxt('Sx.txt',Sxy[0,:,:,0])
   """
   # end_time = time.time()
   # print("--- %s seconds for CNN ---" % (end_time - start_time))
   # print(nn)
   # print(Sxy.shape)
   return Sxy 