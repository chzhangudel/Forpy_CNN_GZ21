#!/bin/env python

import torch
import numpy as np
import importlib
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
def load_paper_net(device: str = 'gpu'):
    """
        Load the neural network from the paper
    """
    print('In load_paper_net()')
    model_module_name = 'subgrid.models.models1'
    model_cls_name = 'FullyCNN'
    model_cls = load_model_cls(model_module_name, model_cls_name)
    print('After load_model_cls()')
    net = model_cls(2, 4)
    print('After net')
    if device == 'cpu':
        transformation = torch.load('/scratch/gpfs/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/final_transformation.pth')
        print('After torch.load()')
    else:
        transformation = pickle_artifact(MODEL_RUN_ID, 'models/transformation')
    net.final_transformation = transformation
    print('After transformation')

    # Load parameters of pre-trained model
    print('Loading the neural net parameters')
    # logging.info('Loading the neural net parameters')
    # client = mlflow.tracking.MlflowClient()
    print('After mlflow.tracking.MlflowClient()')
#    model_file = client.download_artifacts(MODEL_RUN_ID,
#                                           'nn_weights_cpu.pth')
    model_file = '/scratch/gpfs/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/trained_model.pth'
    print('After download_artifacts()')
    if device == 'cpu':
        print('Device: CPU')
        model_file = '/scratch/gpfs/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/nn_weights_cpu.pth'
        net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    else:
        net.load_state_dict(torch.load(model_file))
    print(net)
    return net
nn = load_paper_net('cpu')
nn.eval()

def MOM6_testNN(uv,pe,pe_num,index): 
   global nn,gpu_id
   # start_time = time.time()
   # print('PE number is',pe_num)
   # print('PE is',pe)
   # print('size of uv',uv.shape)
   #normalize the input by 10
#    uv[0,10:-10,10:-10,:] = pe
#    uv[1,10:-10,10:-10,:] = pe+100
#    for j in range(4):
#       for i in range(4):
#           start_row = i * 22 + 10
#           start_col = j * 20 + 10
#           value = i + j * 4
#           uv[0,start_row:start_row+22, start_col:start_col+20,:] = value
#           uv[1,start_row:start_row+22, start_col:start_col+20,:] = value+100
#    if pe==0:
#      np.savetxt('u_pe0.txt',uv[0,:,:,0])
#      np.savetxt('v_pe0.txt',uv[1,:,:,0])
#    if pe==5:
#      np.savetxt('u_pe5.txt',uv[0,:,:,0])
#      np.savetxt('v_pe5.txt',uv[1,:,:,0])
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
   Sxy[0,:,:,:] = (out[0,:,:,:])*scaling
   Sxy[1,:,:,:] = (out[1,:,:,:])*scaling
   Sxy[2,:,:,:] = out[0,:,:,:]*scaling
   Sxy[3,:,:,:] = out[1,:,:,:]*scaling
   Sxy[4,:,:,:] = 1.0/out[2,:,:,:]*scaling
   Sxy[5,:,:,:] = 1.0/out[3,:,:,:]*scaling
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
#    if pe==0:
#      np.savetxt('Sx_pe0.txt',out[0,:,:,0])
#    if pe==5:  
#      np.savetxt('Sx_pe5.txt',out[0,:,:,0])
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
  
#   tensor = torch.zeros(80, 88)
#   for i in range(4):
#       for j in range(4):
#           start_row = i * 20
#           start_col = j * 22
#           value = i * 4 + j 
#           tensor[start_row:start_row+20, start_col:start_col+22] = value
#   print(tensor.shape)
# #   np.savetxt('tensor.txt',tensor)
#   u = torch.zeros(100, 108)
#   v = torch.zeros(100, 108)
#   u[10:90, 10:92] = tensor
#   v[10:90, 10:92] = tensor+100
#   x = torch.zeros(1,2, 100, 100)
#   x[0,0,:,:]=u*10
#   x[0,1,:,:]=v*10
#   print(x.shape)
#   out = nn(x)
#   out = out.detach().numpy().astype(np.float64)
#   out = out.transpose((1,2,3,0)) # new the shape is (4,ni,nj,nk)
#   dim = np.shape(out)
#   print(dim)
#   Sxy = np.zeros((6,dim[1],dim[2],dim[3])) # the shape is (6,ni,nj,nk)
#   epsilon_x = np.random.normal(0, 1, size=(dim[1],dim[2]))
#   epsilon_x = np.dstack([epsilon_x]*dim[3])
#   epsilon_y = np.random.normal(0, 1, size=(dim[1],dim[2]))
#   epsilon_y = np.dstack([epsilon_y]*dim[3])
#   scaling = 1e-7
#   Sxy[0,:,:,:] = (out[0,:,:,:])*scaling
#   Sxy[1,:,:,:] = (out[1,:,:,:])*scaling
#   np.savetxt('Sx_mean.txt',Sxy[0,:,:,0])
#   np.savetxt('Sy_mean.txt',Sxy[1,:,:,0])