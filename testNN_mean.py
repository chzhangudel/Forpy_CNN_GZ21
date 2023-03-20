#!/bin/env python

import numpy as np
import importlib
import math
import time
import matplotlib.pyplot as plt

Sxmean=np.load('/scratch/cimes/cz3321/MOM6/experiments/double_gyre_nonensemble/postprocess/Sxmean.npy')
Symean=np.load('/scratch/cimes/cz3321/MOM6/experiments/double_gyre_nonensemble/postprocess/Symean.npy')
Sxstd=np.load('/scratch/cimes/cz3321/MOM6/experiments/double_gyre_nonensemble/postprocess/Sxstd.npy')
Systd=np.load('/scratch/cimes/cz3321/MOM6/experiments/double_gyre_nonensemble/postprocess/Systd.npy')

Sxmean = Sxmean.transpose((2,1,0)) # new the shape is (ni,nj,nk)
Symean = Symean.transpose((2,1,0)) # new the shape is (ni,nj,nk)
Sxstd = Sxstd.transpose((2,1,0)) # new the shape is (ni,nj,nk)
Systd = Systd.transpose((2,1,0)) # new the shape is (ni,nj,nk)

def MOM6_testNN(uv,pe,pe_num,index): 
   global Sxmean,Symean,Sxstd,Systd
   ids=index[0];ide=index[1];jds=index[2];jde=index[3]
   dim = np.shape(Sxmean[ids-1:ide,jds-1:jde,:])
   out = np.zeros((4,dim[0],dim[1],dim[2])) # the shape is (4,ni,nj,nk)
   out[0,:,:,:] = Sxmean[ids-1:ide,jds-1:jde,:]
   out[1,:,:,:] = Symean[ids-1:ide,jds-1:jde,:]
   out[2,:,:,:] = Sxstd[ids-1:ide,jds-1:jde,:]
   out[3,:,:,:] = Systd[ids-1:ide,jds-1:jde,:]
   Sxy = np.zeros((6,dim[0],dim[1],dim[2])) # the shape is (6,ni,nj,nk)
   epsilon_x = np.random.normal(0, 1, size=(dim[0],dim[1]))
   epsilon_x = np.dstack([epsilon_x]*dim[2])
   epsilon_y = np.random.normal(0, 1, size=(dim[0],dim[1]))
   epsilon_y = np.dstack([epsilon_y]*dim[2])
   Sxy[0,:,:,:] = (out[0,:,:,:] + epsilon_x*out[2,:,:,:])
   Sxy[1,:,:,:] = (out[1,:,:,:] + epsilon_y*out[3,:,:,:])
   Sxy[2,:,:,:] = out[0,:,:,:]
   Sxy[3,:,:,:] = out[1,:,:,:]
   Sxy[4,:,:,:] = out[2,:,:,:]
   Sxy[5,:,:,:] = out[3,:,:,:]
   return Sxy
