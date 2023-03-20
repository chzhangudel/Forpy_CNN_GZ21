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
        transformation = torch.load('/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/final_transformation.pth')
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
    model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/trained_model.pth'
    print('After download_artifacts()')
    if device == 'cpu':
        print('Device: CPU')
        model_file = '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/nn_weights_cpu.pth'
        net.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    else:
        net.load_state_dict(torch.load(model_file))
    print(net)
    return net
nn = load_paper_net('cpu')
nn.eval()
example_input = torch.rand(2, 2, 21, 21)
nn=nn.cuda(0)
example_input=example_input.cuda(0)
module = torch.jit.trace(nn, example_input)
torch.jit.save(module, 'CNN_GPU.pt')
print(module.graph)

def MOM6_testNN(uv,pe,pe_num,index): 
   global module,gpu_id
   # start_time = time.time()
   # print('PE number is',pe_num)
   # print('PE is',pe)
#    print('size of uv',uv.shape)
   #normalize the input by 10
   u = uv[0,:,:,:]*10.0
   v = uv[1,:,:,:]*10.0
#    x = torch.tensor([u,v])
   x = torch.tensor(np.array([u,v]))
#    print('size of x',x.shape)
   x = x.to(torch.float32)
   x = x.transpose(0,3).transpose(1,3).transpose(2,3) # new the shape is (nk,2,ni,nj)
#    print('size of x',x.shape)
   if use_cuda:
       if not next(module.parameters()).is_cuda:
          gpu_id = int(pe/math.ceil(pe_num/torch.cuda.device_count()))
          print('GPU id is:',gpu_id)
          module = module.cuda(gpu_id)
       x = x.cuda(gpu_id)
   with torch.no_grad():
       # start_time = time.time()
       print(x.shape)
       out = module(x)
       # end_time = time.time()
   if use_cuda:
       out = out.to('cpu')
   out = out.to(torch.float64)
#    out = out.numpy().astype(np.float64)
#    print('size of out',out.shape)
   # At this point, python out shape is (nk,4,ni,nj)
   # Comment-out is tranferring arraies into F order
   """
   print(out.shape)
   dim = np.shape(out)
   out = out.flatten(order='F')
   out = out.reshape(dim[0],dim[1],dim[2],dim[3], order='F')
   """
   # convert out to (ni,nj,nk)
   out = out.transpose(2,3).transpose(1,3).transpose(0,3)
#    out = out.transpose((1,2,3,0)) # new the shape is (4,ni,nj,nk)
#    print('size of out',out.shape)
   dim = out.shape
   Sxy = torch.zeros((6,dim[1],dim[2],dim[3])) # the shape is (6,ni,nj,nk)
   epsilon_x = torch.normal(0, 1, size=(dim[1],dim[2]))
#    print('size of epsilon_x',epsilon_x.shape)
   epsilon_x = torch.dstack([epsilon_x]*dim[3])
#    print('size of epsilon_x',epsilon_x.shape)
   epsilon_y = torch.normal(0, 1, size=(dim[1],dim[2]))
   epsilon_y = torch.dstack([epsilon_y]*dim[3])

   epsilon_x = torch.normal(torch.zeros(dim[1],dim[2]),torch.ones(dim[1],dim[2]))
   epsilon_x = torch.dstack([epsilon_x]*dim[3])
   epsilon_y = torch.normal(torch.zeros(dim[1],dim[2]),torch.ones(dim[1],dim[2]))
   epsilon_y = torch.dstack([epsilon_y]*dim[3])

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
   Sxy[0,:,:,:] = (out[0,:,:,:] + epsilon_x/out[2,:,:,:])*scaling
   Sxy[1,:,:,:] = (out[1,:,:,:] + epsilon_y/out[3,:,:,:])*scaling
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
   # end_time = time.time()
   # print("--- %s seconds for CNN ---" % (end_time - start_time))
   # print(nn)
#    print('size of epsilon_x',Sxy.shape)
#    Sxy = Sxy.tolist()
   Sxy = Sxy.numpy().astype(np.float64)
   return Sxy

gpu_id=0
uv = np.random.rand(2, 40, 40, 4)
Sxy=MOM6_testNN(uv,1,1,1)
print(Sxy.shape)