from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import math

BOX_COLOR1c = (255,0,144) #hot pink
BOX_COLOR1 = (1.,0.,0.565) #hot pink
BOX_COLOR2 = (0.663,0.51,1.) #bright purple


def stacked_bbox_iou(boxes1, boxes2):
    """
    bboxes1 and boxes2 has the same dimension, which is [num cells, 4]
    Returns the IoUs of two sets of bounding boxes   
    """
    #Get the coordinates of bounding boxes
    xc1, yc1, width1, height1 = boxes1[:,0], boxes1[:,1], boxes1[:,2], boxes1[:,3]
    xc2, yc2, width2, height2 = boxes2[:,0], boxes2[:,1], boxes2[:,2], boxes2[:,3]
    
    b1_x1 = xc1 -0.5*width1
    b1_x2 = xc1 +0.5*width1
    b1_y1 = yc1 -0.5*height1
    b1_y2 = yc1 +0.5*height1
    
    b2_x1 = xc2 -0.5*width2
    b2_x2 = xc2 +0.5*width2
    b2_y1 = yc2 -0.5*height2
    b2_y2 = yc2 +0.5*height2
    
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
    
    ious = inter_area / (b1_area + b2_area - inter_area)
    
    return ious

def find_max_iou(bbox_gt, bboxes, has_obj_mask, CUDA = True):
    """
    bboxes is a stacked of candidate bounding boxes derived from the anchors,
    its dimension is [num cells, num anchors, 4]
    
    bbox_gt is the ground truth bounding box,
    its dimension is [num cells, 4]
    
    has_obj_mask indicates which cell has an object
    
    Returns the candidate bounding box that has the highest iou with the ground truth
    """
    
    max_iou = torch.zeros([len(has_obj_mask)]).cuda()
    curr_iou = torch.zeros([len(has_obj_mask)]).cuda()
    max_ind =(-1)*torch.ones([len(has_obj_mask)]).cuda()
    
    if (has_obj_mask.sum() == 0):
        return max_ind, max_iou
    
    max_iou[has_obj_mask] = stacked_bbox_iou(bbox_gt[has_obj_mask], bboxes[has_obj_mask,0,:4])
    max_ind[has_obj_mask] = 0
    
    for i in range(1,bboxes.size(1)):
        curr_iou[has_obj_mask] = stacked_bbox_iou(bbox_gt[has_obj_mask], bboxes[has_obj_mask,i,:4])
        gt_mask = torch.gt(curr_iou, max_iou) & has_obj_mask
        
        max_ind = max_ind + gt_mask.int()
        max_iou = max_iou*((~gt_mask).int()) + gt_mask.int()*curr_iou
    
    return max_ind, max_iou      


def find_center_grid(bbox, stride):
    """
    Given a bounding box in its actual dimension and stride,
    find out which grid cell its center is located in
    """
    xmin, ymin, width, height = bbox
    xcenter = xmin+0.5*width
    ycenter = ymin+0.5*height
    
    return int(xcenter//stride), int(ycenter//stride)


def ground_truth_transform(bboxes, inp_dim, stride):
    """
    Transform the list(dict) of ground truth bounding boxes into a tensor,
    which will be used as an input to yolo_loss3(). 
    
    Original ground truth box is in (xmin, ymin, w, h)
    Output ground truth box is in (xc, yc, w, h)

    Output dimension is [batch_size, num cell, 4]
    """ 
    batch_size = len(bboxes)
    grid_size = inp_dim //stride
    
    ground_truth = torch.zeros([batch_size, grid_size, grid_size, 4])
    num_bboxes = torch.zeros([batch_size])
    
    for i in range(batch_size):
        
        for j, bbox in enumerate(bboxes[i]):
            
            xmin, ymin, width, height = bbox
            
            xc = xmin+0.5*width
            yc = ymin+0.5*height
            
            xgrid,ygrid = find_center_grid(bbox, stride)    
            
            ground_truth[i, ygrid, xgrid, :4] = torch.tensor([xc,yc,width,height])
    
    ground_truth = ground_truth.view(batch_size, grid_size*grid_size, 4)
    
    return ground_truth

def check_ground_truth_grid(ground_truth,stride):
    """
    Just for debugging 
    """    
    
    grid_size = int(math.sqrt(ground_truth.size(1)))
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    x_y_offset = torch.cat((x_offset, y_offset), 1)*stride
  
    result = torch.cat((x_y_offset, ground_truth[0,:,:2]),1)
    
    return result
   

def yolo_loss(inp_dim, num_anchors, prediction, ground_truth, a1 = 1, b1 =1, c1 = 1, d1 = 1, verbose = False, CUDA = True):
    """
    Calculate detection loss.
    
    Loss has 4 components: bounding box center, bounding box width and height, objectness confidence, no object penalty
    
    a1, b1, c1, d1 are weights of these 4 components
    
    bbox is in (xc, yc, width, height)
    """
   
    batch_size = prediction.size(0)
    total_grid = prediction.size(1)//num_anchors
    box_attr_length = prediction.size(2)
   
    prediction = prediction.view(batch_size, total_grid, box_attr_length*num_anchors)
    
    total_loss = 0
                  
    for i in range(batch_size):
        loss = torch.zeros([total_grid])
        num_bboxes = torch.zeros([total_grid])
    
        if CUDA:
            ground_truth = ground_truth.cuda()
            loss = loss.cuda()
            num_bboxes = num_bboxes.cuda()

        bbox = ground_truth[i ,:, :4]
        
        has_obj_mask = torch.gt(ground_truth[i,:,2],0)
             
        # Find which predicted box has the highest iou with the ground truth box    
        max_ind, max_iou = find_max_iou(bbox, prediction[i,:,:].view(total_grid, num_anchors, box_attr_length), has_obj_mask)
          
        for m in range(num_anchors):
            
            # We will only penalize the bounding box errors and objectness confidence error 
            # of the predicted box that has the highest iou with the ground truth box
            bbox_mask = torch.eq(max_ind, m)                       
            o_b_mask = (has_obj_mask & bbox_mask)
                     
            loss[o_b_mask] += a1*((prediction[i,o_b_mask,m*box_attr_length]/inp_dim-bbox[o_b_mask, 0]/inp_dim)**2 + 
                                             (prediction[i,o_b_mask,m*box_attr_length+1]/inp_dim-bbox[o_b_mask,1]/inp_dim)**2)
                  
            loss[o_b_mask] += b1*((torch.sqrt(prediction[i,o_b_mask,m*box_attr_length+2]/inp_dim)-torch.sqrt(bbox[o_b_mask,2]/inp_dim))**2 +
                                     (torch.sqrt(prediction[i,o_b_mask,m*box_attr_length+3]/inp_dim)-torch.sqrt(bbox[o_b_mask,3]/inp_dim))**2)
           
            loss[o_b_mask] += c1*((prediction[i,o_b_mask,m*box_attr_length+4]-1.)**2)
         
            loss[~has_obj_mask] += d1*((prediction[i, ~has_obj_mask, 4+m*box_attr_length])**2)
                                          
        total_loss += loss.sum()    
        if verbose:    
            print(has_obj_mask.cpu().numpy())    
            print(loss.cpu().detach().numpy())            
                      
    return total_loss   

def visualize_detector_output(output, data, img_dir, img_only_transform):
    """
    create a dictionary, the key is the row index in the data table (each row is one image), 
    the value is a tensor that each row is a bound box prediction
    its dimension is [num predicted boxes, 5]
    
    bbox is in (xmin, ymin, xmax, ymax)
    
    """
    for table_index in output:
           
        x = output[table_index]
        filename = img_dir+data.iloc[table_index]['image_id']
    
        mat = cv2.imread(filename)      
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

        augmented = img_only_transform(image = mat)

        mat = augmented['image']
        x[:,2] = x[:,2] - x[:,0]
        x[:,3] = x[:,3] - x[:,1]
      
        visualize(mat, x[:,:4])     
        
def visualize_transformed_ground_truth(mat, bboxes, bboxes2):
    """
    visualize the ground truth and transformed ground truth on ONE image
    
    bboxes are the original ground truth      
    bboxes2 are the transformed ground truth
       
    bboxes is in (xmin, ymin, w, h)
    boxes2 is in (xc, yc, w, h)  
    """
    
    mat = cv2.UMat(mat)
    
    TEXT_COLOR = (255, 255, 255)
        
    for i, bbox in enumerate(bboxes):

        x_min, y_min, w, h = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
        
        cv2.rectangle(mat, (x_min, y_min), (x_max, y_max), color=BOX_COLOR1, thickness=1)

    for i in range(bboxes2.size(0)):

        x_c, y_c, w, h = bboxes2[i,:]
        
        if (w == 0 or h ==0):
            continue
            
        x_min, x_max, y_min, y_max = int(x_c-0.5*w), int(x_c + 0.5*w), int(y_c - 0.5*h), int(y_c + 0.5 *h)

        cv2.rectangle(mat, (x_min, y_min), (x_max, y_max), color=BOX_COLOR2, thickness=1)

    plt.figure(figsize=(12, 12))
    plt.imshow(mat.get()) 

def visualize(mat, bboxes):
    """
    visualize predicted bounding boxes
     
    bboxes is in (xmin, ymin, w, h)
    """
    
    mat = cv2.UMat(mat)
    
    TEXT_COLOR = (255, 255, 255)
          
    for i in range(bboxes.size(0)):

        x_min, y_min, w, h = bboxes[i,:]
        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

        cv2.rectangle(mat, (x_min, y_min), (x_max, y_max), color=BOX_COLOR1c, thickness=1)

    plt.figure(figsize=(12, 12))
    plt.imshow(mat.get())    
    
def init_detector_weights(model):
    
    for name, param in model.named_parameters():
                
        if "DT" in name:
            if "conv_" in name and "weight" in name:        
                torch.nn.init.xavier_uniform_(param)
                param.requires_grad = True
            elif "conv" in name and "bias" in name:   
                torch.nn.init.constant_(param,0.5)
                param.requires_grad = True
            elif "batch_norm" in name and "weight" in name:   
                torch.nn.init.constant_(param,1)
                param.requires_grad = True
            elif "batch_norm" in name and "bias" in name:   
                torch.nn.init.constant_(param,0)
                param.requires_grad = True    
        else: 
            param.requires_grad = False
    
    return model

def set_DT_requires_grad_only(model):
    
    for name, param in model.named_parameters():
                
        if "DT" in name:
            param.requires_grad = True    
        else: 
            param.requires_grad = False
    
    return model