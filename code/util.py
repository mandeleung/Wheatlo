from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes   
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, CUDA = True):

    #we can think of this as converting from pixel space to stride space, similar to changing the unit, say from cm to inch
    #say an image of 20 px width, if we convert to a stride space of 5 pixel, then the image has a width of 4 strides
    
    #if the input image has a width of 608 px, and the input feature map's width at this layer is 19,
    #that means each stride has a width of (608/19) 32 pixels, and there are 19 strides.
  
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2) #original img width / input feature map width; height is just the same

    grid_size = inp_dim // stride # number of grid cells along the width; height is just the same
    bbox_attrs = 5 
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    #achors were given in pixel space, convert them to stride space
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        
    #print(torch.cat((x_offset, y_offset), 1))

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,:4] *= stride
    
    return prediction

def write_results(prediction, confidence, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
     
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)
    
    hasOutput = False
 
    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
       
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        
        if len(non_zero_ind) > 0:         
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,5)
            
        else:
            continue
            
        conf_sort_index = torch.sort(image_pred_[:,4], descending = True )[1] #[0] is the value, [1] is the index
        image_pred_ = image_pred_[conf_sort_index]

        idx = image_pred_.size(0)   #Number of detections
        i = 0;
        while i+1 < idx: 
            #Get the IOUs of all boxes that come after the one we are looking at 
            #in the loop
            ious = bbox_iou(image_pred_[i].unsqueeze(0), image_pred_[i+1:])                   

            #Zero out all the detections that have IoU > treshhold
            iou_mask = (ious < nms_conf).float().unsqueeze(1)
            image_pred_[i+1:] *= iou_mask       

            #Remove the zero entries
            non_zero_ind = torch.nonzero(image_pred_[:,4])

            if (len(non_zero_ind) == 0):
                break

            non_zero_ind = non_zero_ind.squeeze()
           
            image_pred_ = image_pred_[non_zero_ind].view(-1,5)

            idx = image_pred_.size(0)
            i = i+1    
            
        batch_ind = image_pred_.new(image_pred_.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
        seq = batch_ind, image_pred_
            
        if not hasOutput:
            output = torch.cat(seq,1)
            hasOutput = True
        else:
            out = torch.cat(seq,1)
            output = torch.cat((output,out))    
                
    if hasOutput:
        return output
    else:
        return 0
    
