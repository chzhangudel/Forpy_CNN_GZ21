#!/bin/env python

import torch
import numpy as np
import importlib
import math
import time
from torch.nn import Parameter

REPO = 'subgrid'  #'gz21'

# GPU setup
args_no_cuda = False #True when manually turn off cuda
use_cuda = not args_no_cuda and torch.cuda.is_available()
if use_cuda:
    print('device for inference on',torch.cuda.device_count(),'GPU(s)')
else:
    print('device for inference on CPU')

#load the neural network
def load_model_cls(model_module_name: str, model_cls_name: str):
    try:
        module = importlib.import_module(model_module_name)
        model_cls = getattr(module, model_cls_name)
    except ModuleNotFoundError as e:
        raise type(e)('Could not retrieve the module in which the trained model \
                      is defined: ' + str(e))
    except AttributeError as e:
        raise type(e)('Could not retrieve the model\'s class. ' + str(e))
    return model_cls
def load_paper_net(device: str = 'gpu', model_cls_name: str = 'FullyCNN'):
    """
        Load the neural network from the paper
    """
    print('In load_paper_net()')
    model_module_name = f'{REPO}.models.models1'
    # model_cls_name = 'CNN15x15'
    model_cls = load_model_cls(model_module_name, model_cls_name)
    print('After load_model_cls()')
    net = model_cls(2,4,batch_norm=1)
    
    # final_transform= '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/final_transformation_04292023.pth'
    # print('After net')
    # if device == 'cpu':
    #     transformation = torch.load(final_transform)
    #     print('After torch.load()')
    # else:
    #     transformation = pickle_artifact(MODEL_RUN_ID, 'models/transformation')
    # net.final_transformation = transformation
    print('After transformation')
    # Load parameters of pre-trained model
    print('After mlflow.tracking.MlflowClient()')
    
    
    # ----------------- CHANGE THIS PATH TO TRAINED MODEL ----------------- #
    model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/trained_model_cnn15x15_landmask_none.pth'
    # ---------------------------------------------------- #
    
    
    print('Loading final transformation')
    model_module_name = f'{REPO}.models.transforms'
    model_cls_name = 'SoftPlusTransform'
    model_cls = load_model_cls(model_module_name, model_cls_name)
    model_cls_name = 'PrecisionTransform'
    model_cls1 = load_model_cls(model_module_name, model_cls_name)
    transform = model_cls.__new__(model_cls,)
    model_cls1.__init__(transform,)
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    transform._min_value = Parameter(state_dict.pop('final_transformation._min_value'))
    transform.indices = slice(2,4)
    print('After download_artifacts()')
    net.load_state_dict(state_dict)
    net.final_transformation = transform
    print(net)
    return net
nn = load_paper_net('cpu',model_cls_name = 'CNN15x15')
nn.eval()

def MOM6_testNN(uv,pe,pe_num,index): 
   global nn,gpu_id
#    start_time = time.time()
   # print('PE number is',pe_num)
   # print('PE is',pe)
   # print('size of uv',uv.shape)

   #set boundary condition
# #    print('index:', index)
# #    print('uv shape:', np.shape(uv))
# #    np.savetxt(f'uv{pe}.txt',(uv[0,:,:,0]))
#    landmask = np.ones(np.shape(uv)) 
#    halo=7
# #    print(index)
#    if index[0]==1:
#        landmask[:,:halo,:,:]=np.nan
#    if index[1]==88:
#        landmask[:,-halo:,:,:]=np.nan
#    if index[2]==1:
#        landmask[:,:,:halo,:]=np.nan
#    if index[3]==80:
#        landmask[:,:,-halo:,:]=np.nan
#    uv = uv*landmask
# #    np.savetxt(f'landmask{pe}.txt',(landmask[0,:,:,0]))
# #    import sys
# #    sys.exit(0)

   #normalize the input by 10
   u = uv[0,:,:,:]*10.0
   v = uv[1,:,:,:]*10.0
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
    #    start_time1 = time.time()
       out = nn(x)
    #    end_time1 = time.time()
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
#    print('output shape:',dim)
   Sxy = np.zeros((6,dim[1],dim[2],dim[3])) # the shape is (6,ni,nj,nk)
   epsilon_x = np.random.normal(0, 1, size=(dim[1],dim[2]))
   epsilon_x = np.dstack([epsilon_x]*dim[3])
   epsilon_y = np.random.normal(0, 1, size=(dim[1],dim[2]))
   epsilon_y = np.dstack([epsilon_y]*dim[3])
   scaling = 1e-7
   # if pe==0:
   #   print(scaling)
   # mean output
   """
   Sxy[0,:,:,:] = (out[0,:,:,:])*scaling
   Sxy[1,:,:,:] = (out[1,:,:,:])*scaling
   # std output
   Sxy[0,:,:,:] = (epsilon_x/out[2,:,:,:])*scaling
   Sxy[1,:,:,:] = (epsilon_y/out[3,:,:,:])*scaling
   """
   # full output
#    Sxy[0,:,:,:] = (out[0,:,:,:] + epsilon_x/out[2,:,:,:])*scaling
#    Sxy[1,:,:,:] = (out[1,:,:,:] + epsilon_y/out[3,:,:,:])*scaling
#    Sxy[2,:,:,:] = out[0,:,:,:]*scaling
#    Sxy[3,:,:,:] = out[1,:,:,:]*scaling
#    Sxy[4,:,:,:] = 1.0/out[2,:,:,:]*scaling
#    Sxy[5,:,:,:] = 1.0/out[3,:,:,:]*scaling
   Sxy[0,:,:,:] = (out[0,:,:,:] )*scaling
   Sxy[1,:,:,:] = (out[1,:,:,:] )*scaling
   Sxy[2,:,:,:] = 0.0
   Sxy[3,:,:,:] = 0.0
   Sxy[4,:,:,:] = 0.0
   Sxy[5,:,:,:] = 0.0
   """
   # scaling the parameters for upper and lower layers
   Sxy[:,:,:,0]=Sxy[:,:,:,0]*0.8
   Sxy[:,:,:,1]=Sxy[:,:,:,1]*1.5
   """
   """
   np.savetxt('Sx_mean.txt',Sxy[2,:,:,0])
   np.savetxt('Sy_mean.txt',Sxy[3,:,:,0])
   np.savetxt('Sx_std.txt',Sxy[4,:,:,0])
   np.savetxt('Sy_std.txt',Sxy[5,:,:,0])
   np.savetxt('WH_u.txt',uv[0,:,:,0])
   np.savetxt('WH_v.txt',uv[1,:,:,0])
   """
#    end_time = time.time()
#    print("--- %s seconds for CNN ---" % (end_time1 - start_time1))
#    print("--- %s seconds for total ---" % (end_time - start_time))
   # print(nn)
   # print(Sxy.shape)
   return Sxy

if __name__ == '__main__':
#   start_time = time.time()
  x = np.zeros((1, 2, 24, 24)).astype(np.float32)
#   x[:,1,:12, :] = 1.0*10
#   print(x[0,1,:,:])
  x = torch.from_numpy(x)
  if use_cuda:
      if not next(nn.parameters()).is_cuda:
         gpu_id = int(pe/math.ceil(pe_num/torch.cuda.device_count()))
         print('GPU id is:',gpu_id)
         nn = nn.cuda(gpu_id)
      x = x.cuda(gpu_id)
  with torch.no_grad():
   #    start_time1 = time.time()
      out = nn(x)
   #    end_time1 = time.time()
  if use_cuda:
      out = out.to('cpu')
  out = out.numpy().astype(np.float64)
  out = out.transpose((1,2,3,0)) # new the shape is (4,ni,nj,nk)
  dim = np.shape(out)
  scaling = 1e-7
  np.savetxt('/scratch/cimes/cz3321/MOM6/experiments/double_gyre_nonensemble/postprocess/zero_input/Sx_mean_model5.txt',(out[0,:,:,0])*scaling)
  np.savetxt('/scratch/cimes/cz3321/MOM6/experiments/double_gyre_nonensemble/postprocess/zero_input/Sy_mean_model5.txt',(out[1,:,:,0])*scaling)