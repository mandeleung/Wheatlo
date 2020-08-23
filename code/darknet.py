from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * 

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    print(f"Total number of lines in file: {len(lines)}")
    
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    
    prefix = "FE"                                         # layers that are part of the feature extractor, darknet-53
    
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == '#':
            if "Detector Begins" in line:
                prefix = "DT"
                continue
            elif "End" in line:
                break
            else:
                continue
        
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip() 
            block["purpose"] = prefix
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
     
    module_list = nn.ModuleList()
    prev_filters = 3 # input depth
    output_filters = [] # output depth of all blocks
    
    for index, x in enumerate(blocks):
        module = nn.Sequential()
       
        #check the type of block
        #create a new module for the block
        #append to module_list
        #append filters to output_fitlers to keep track of output depth
        
        prefix = x["purpose"]
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            if "batch_normalize" in x:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            else:
                batch_normalize = 0
                bias = True
        
            filters= int(x["filters"]) #output depth
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernel_size - 1) // 2 #padding is always half the kernel size or zero
            else:
                pad = 0
        
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module(f"{prefix}conv_{index}", conv)
        
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"{prefix}batch_norm_{index}", bn)
        
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True) #like ReLU but not strickly zero on the -ve side
                module.add_module(f"{prefix}leaky_{index}", activn)
        
            #If it's an upsampling layer
            #We use Bilinear2dUpsampling #but the code said "nearest"??
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module(f"{prefix}upsample_{index}", upsample)
            #no need to re-assign filters because input depth same as output depth
                
        #If it is a route layer
        elif (x["type"] == "route"): 
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            if len(x["layers"]) > 1:
                end = int(x["layers"][1])
            else:
                end = 0
            #Turn absolution pos to relative pos, then it doesn't matter if it is zero-based or one-based
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module(f"{prefix}route_{index}", route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
    
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module(f"{prefix}shortcut_{index}", shortcut)
            #no need to re-assign filters because input depth same as output depth
            
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",") #mask indicates which anchor box to use
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)
            module.add_module(f"{prefix}Detection_{index}", detection)
                              
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (module_list, output_filters)

def append_modules(blocks, module_list, output_filters):
     
    prev_filters = output_filters[-1] # input depth
    
    for index, x in enumerate(blocks):
        module = nn.Sequential()
       
        #check the type of block
        #create a new module for the block
        #append to module_list
        #append filters to output_fitlers to keep track of output depth
        
        prefix = x["purpose"]
        #If it's a convolutional layer
        if (x["type"] == "convolutional"):
            #Get the info about the layer
            activation = x["activation"]
            if "batch_normalize" in x:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            else:
                batch_normalize = 0
                bias = True
        
            filters= int(x["filters"]) #output depth
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
        
            if padding:
                pad = (kernel_size - 1) // 2 #padding is always half the kernel size or zero
            else:
                pad = 0
        
            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module(f"{prefix}conv_{index}", conv)
        
            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"{prefix}batch_norm_{index}", bn)
        
            #Check the activation. 
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True) #like ReLU but not strickly zero on the -ve side
                module.add_module(f"{prefix}leaky_{index}", activn)
        
            #If it's an upsampling layer
            #We use Bilinear2dUpsampling #but the code said "nearest"??
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module(f"{prefix}upsample_{index}", upsample)
            #no need to re-assign filters because input depth same as output depth
                
        #If it is a route layer
        elif (x["type"] == "route"): 
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            if len(x["layers"]) > 1:
                end = int(x["layers"][1])
            else:
                end = 0
            #Turn absolution pos to relative pos, then it doesn't matter if it is zero-based or one-based
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module(f"{prefix}route_{index}", route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
    
        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module(f"{prefix}shortcut_{index}", shortcut)
            #no need to re-assign filters because input depth same as output depth
            
        #Yolo is the detection layer
        elif x["type"] == "yolo":
            mask = x["mask"].split(",") #mask indicates which anchor box to use
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)
            module.add_module(f"{prefix}Detection_{index}", detection)
                              
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (module_list, output_filters)

class Darknet(nn.Module):
    def __init__(self, cfgfile, inp_dim, weightfile = None):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.inp_dim = inp_dim
        self.module_list, self.output_filters = create_modules(self.blocks)
        if weightfile:
            self.load_weights(weightfile)
        
    def add_extractor_layers(self):
        
        glb_avg_block = {'type': 'globalAvgPool','purpose': 'FE'}
        self.blocks.append(glb_avg_block)
        
        module = nn.Sequential()
        glbAvgPool = EmptyLayer()
        module.add_module(f"FEglobalAvgPool_{len(self.module_list)}", glbAvgPool)
        self.module_list.append(module)
         
        fc_block = {'type': 'fullyConnected','outputs' : 2,'purpose': 'FE'}
        self.blocks.append(fc_block)
        
        module = nn.Sequential()
        fc = nn.Linear(self.output_filters[-1], 2)
        module.add_module(f"FEfullyConnected_{len(self.module_list)}", fc)
        self.module_list.append(module)
        
    def remove_extractor_layers(self):
        """
        Create a new module list, then copy over the original list but minus the last two modules
        """
        
        module_list = nn.ModuleList()
        blocks = []
        
        for i in range(len(self.module_list)-2):
            module_list.append(self.module_list[i])
            blocks.append(self.blocks[i])
        
        self.module_list = module_list
        self.blocks = blocks
        
        return      
             
    def add_detector_layers(self, cfgfile):
        
        detector_blocks = parse_cfg(cfgfile)
        
        append_modules(detector_blocks, self.module_list, self.output_filters)
            
        self.blocks = self.blocks + detector_blocks    
        
        return
        
    def forward(self, x):
        modules = self.blocks
        outputs = {}   #We cache the outputs for the route layer
        
        write = 0
        for i, module in enumerate(modules):        
            module_type = (module["type"])
            
            if module_type == "convolutional" or module_type == "upsample" or module_type == "fullyConnected":
                x = self.module_list[i](x)
    
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
    
            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
    
            elif module_type == "globalAvgPool":
            
                x = torch.mean(torch.mean(x, 3),2) # Batch_size X depth X W X H, average each feature map
            
            elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                
                #Transform 
                x = predict_transform(x, self.inp_dim, anchors)
                return x
                    
            outputs[i] = x
        
        return x


    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        print(self.header)
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        
        for i in range(len(self.module_list)):
            
            module_type = self.blocks[i]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i] # model would be a nn.sequential
                
                batch_normalize = int(self.blocks[i]["batch_normalize"])                
                conv = model[0]             
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


