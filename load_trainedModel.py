import numpy as np
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

'''
'ResNet50',
'ResNet50_v2',
'MobileNet_v3_large',
'MobileNet_v3_large_320'
'''

# function to get the base model for each backbone
def get_model_instance_segmentation(num_classes, backbone):
    if backbone == 'ResNet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    
    elif backbone == 'ResNet50_fpn_v2':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    
    elif backbone == 'MobileNet_v3_large_fpn':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    
    elif backbone == 'MobileNet_v3_large_320_fpn':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

# function to load the trained model
def get_trained_model_v01(backbone, device): #(home_dir, backbone, device):
    model = get_model_instance_segmentation(7, backbone)
    model.to(device)
    
    resnet50 = 'https://github.com/Hakim-0000/app_satu/releases/download/v0.1/model_40_frcnn_fpnv1.pt'
    resnet50v2 = 'https://github.com/Hakim-0000/app_satu/releases/download/v0.1/model_40_frcnn_fpnv2.pt'
    mobnetv3 = 'https://github.com/Hakim-0000/app_satu/releases/download/v0.1/model_40_frcnn_mobnet_large_fpn.pt'
    mobnetv3_320 = 'https://github.com/Hakim-0000/app_satu/releases/download/v0.1/model_40_frcnn_mobnet_large_320fpn.pt'
    
    resnet50_state_dict = torch.hub.load_state_dict_from_url(resnet50, map_location=torch.device('cpu'))
    resnet50v2_state_dict = torch.hub.load_state_dict_from_url(resnet50v2, map_location=torch.device('cpu'))
    mobnetv3_state_dict = torch.hub.load_state_dict_from_url(mobnetv3, map_location=torch.device('cpu'))
    mobnetv3_320_state_dict = torch.hub.load_state_dict_from_url(mobnetv3_320, map_location=torch.device('cpu'))
    
    if backbone == 'ResNet50_fpn':
        model.load_state_dict(resnet50_state_dict)
        
    elif backbone == 'ResNet50_fpn_v2':
        model.load_state_dict(resnet50v2_state_dict)
        
    elif backbone == 'MobileNet_v3_large_fpn':
        model.load_state_dict(mobnetv3_state_dict)
        
    elif backbone == 'MobileNet_v3_large_320_fpn':
        model.load_state_dict(mobnetv3_320_state_dict)
    
    return model

# function to load the trained model
def get_trained_model_v02(backbone, device): #(home_dir, backbone, device):
    model = get_model_instance_segmentation(7, backbone)
    model.to(device)
    
    resnet50 = 'https://github.com/Hakim-0000/app_satu/releases/download/v0.2/retrained_80_resnet50v1.pt'
    resnet50v2 = 'https://github.com/Hakim-0000/app_satu/releases/download/v0.2/retrained_80_resnet50v2.pt'
    mobnetv3 = 'https://github.com/Hakim-0000/app_satu/releases/download/v0.2/retrained_80_mobnetv3_L.pt'
    mobnetv3_320 = 'https://github.com/Hakim-0000/app_satu/releases/download/v0.2/retrained_80_mobnetv3_L_320.pt'
    
    resnet50_state_dict = torch.hub.load_state_dict_from_url(resnet50, map_location=torch.device('cpu'))
    resnet50v2_state_dict = torch.hub.load_state_dict_from_url(resnet50v2, map_location=torch.device('cpu'))
    mobnetv3_state_dict = torch.hub.load_state_dict_from_url(mobnetv3, map_location=torch.device('cpu'))
    mobnetv3_320_state_dict = torch.hub.load_state_dict_from_url(mobnetv3_320, map_location=torch.device('cpu'))
    
    if backbone == 'ResNet50_fpn':
        model.load_state_dict(resnet50_state_dict)
        
    elif backbone == 'ResNet50_fpn_v2':
        model.load_state_dict(resnet50v2_state_dict)
        
    elif backbone == 'MobileNet_v3_large_fpn':
        model.load_state_dict(mobnetv3_state_dict)
        
    elif backbone == 'MobileNet_v3_large_320_fpn':
        model.load_state_dict(mobnetv3_320_state_dict)
    
    return model
        

def load_models(trained_version, backbone, device):
    if trained_version == 'v01':
        models = get_trained_model_v01(backbone, device)
    
    elif trained_version == 'v02':
        models = get_trained_model_v02(backbone, device)
    
    return models

