#!/bin/env python

import torch
import numpy as np
import subgrid
import math
import time

# GPU setup
args_no_cuda = False #True when manually turn off cuda
use_cuda = not args_no_cuda and torch.cuda.is_available()
if use_cuda:
    print('device for inference on',torch.cuda.device_count(),'GPU(s)')
else:
    print('device for inference on CPU')

#load the neural network
nn = subgrid.load_paper_net('cpu')
nn.eval()

def MOM6_testNN(u,v,pe,pe_num): 
   global nn,gpu_id
   # start_time = time.time()
   # print('PE number is',pe_num)
   # print('PE is',pe)
   # print(u.shape,v.shape)
   #normalize the input by 10
   u = u*10.0
   v = v*10.0
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
       out = nn(x)
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
   scaling = 1e-7
   # if pe==0:
   #   print(scaling)
   # mean output
   Sxy[0,:,:,:] = (out[0,:,:,:])*scaling
   Sxy[1,:,:,:] = (out[1,:,:,:])*scaling
   """
   # std output
   Sxy[0,:,:,:] = (epsilon_x/out[2,:,:,:])*scaling
   Sxy[1,:,:,:] = (epsilon_y/out[3,:,:,:])*scaling
   # full output
   Sxy[0,:,:,:] = (out[0,:,:,:] + epsilon_x/out[2,:,:,:])*scaling
   Sxy[1,:,:,:] = (out[1,:,:,:] + epsilon_y/out[3,:,:,:])*scaling
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

# if __name__ == '__main__':
#   start_time = time.time()
#   x = np.arange(1, 1251).astype(np.float32)
#   x = x / 100
#   print(x[:10])
#   x = x.reshape((1, 2, 25, 25), order='F')
#   x = torch.tensor(x)
#   if use_cuda:
#       x = x.to(device)
#   with torch.no_grad():
#       out = nn(x)
#   if use_cuda:
#       out = out.to('cpu')
#   out = out.numpy()
#   out = out.flatten(order='F')
#   print("BEGINNING OF PYTHON")
#   print(out[:10])
#   print("END OF PYTHON")
#   end_time = time.time()
#   print("time elapse with", device, "is", end_time-start_time, "s")