import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)

    # # fiter out invalid depth
    # import pdb; pdb.set_trace()
    # mask = depthmap > 1.25
    # mask = mask.reshape(-1)
    # points = points[mask]
    # depthmap = depthmap.reshape(-1)[mask]

    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o

    # points_np = points.cpu().detach().numpy()
    # np.savetxt('1.txt', points_np, fmt='%.6f', delimiter=' ')
    # import pdb; pdb.set_trace()

    return points

def depth_to_normal_geo(depth, k = 15):  
    # input: tensor(b,3,h,w){, int(knn)}
    # output: tensor(b,3,h,w)
    B,C,H,W = depth.shape
    point_martx = torch.nn.functional.unfold(depth, (k, k), dilation=1, padding=int((k-1)/2), stride=1) # b,(3*k*k),h*w 
    matrix_a = point_martx.transpose(1,2).reshape(B,H,W,C,k*k).transpose(-1,-2)  #b,h,w,k*k,3
    matrix_a_zero = torch.zeros_like(matrix_a, dtype=torch.float16)

    matrix_a_trans = matrix_a.transpose(-1,-2) #b,h,w,3,k*k
    matrix_b = torch.ones([B, H, W, k * k, 1])

    point_multi = torch.matmul(matrix_a_trans, matrix_a) #b,h,w,3,3
    inv_matrix = torch.linalg.inv(point_multi) #b,h,w,3,3 ###### torch>1.8 -> torch.linalg.inv
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans),matrix_b) #b,h,w,3,1
    normals = generated_norm.squeeze(-1).transpose(2,3).transpose(1,2) #b,3,h,w

    return normals

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3) #h,w,3
    # mask = depth > 0
    # mask = mask.permute(1,2,0).float() #h,w,1
    # points = points*mask

    # points_np = points.reshape(-1, 3)
    # points_np = points_np.cpu().detach().numpy()
    # np.savetxt('2.txt', points_np, fmt='%.6f', delimiter=' ')
    # import pdb; pdb.set_trace()
    
    # from 2D GS
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map

    # from mine
    # import pdb; pdb.set_trace()
    # points_tensor = points.unsqueeze(0).permute(0,3,1,2)#1,3,h,w
    # output = depth_to_normal_geo(points_tensor, k=15)
    # output = (output +1)/2

    return output