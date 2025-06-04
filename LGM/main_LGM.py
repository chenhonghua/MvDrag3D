import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import cv2

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from LGM.core.options import AllConfigs, config_defaults
from LGM.core.models import LGM

from PIL import Image
from torchvision.utils import save_image

def draw_lines_on_subimage(subimage, points):
    for i in range(0, points.shape[1], 2):
        start_point = (points[0, i], points[1, i])
        end_point = (points[0, (i + 1) % points.shape[1]], points[1, (i + 1) % points.shape[1]])
        cv2.line(subimage, start_point, end_point, color=(0, 0, 255), thickness=2)
    return subimage

# process function
def processLGM(input_image: np.ndarray,
               model_path: str,
               workspace_name: str,
               isProject2D: bool = False,
               src_points_path: str = None,
               tgt_points_path: str = None,
               stage: str = 'initial',
               device: str = 'cuda:0'
               ):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    # load some fixed options for LGM
    opt = config_defaults['big']

    # model
    model = LGM(opt)

    # resume pretrained checkpoint
    if model_path is not None:
        if model_path.endswith('safetensors'):
            ckpt = load_file(model_path, device='cpu')
        else:
            ckpt = torch.load(model_path, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        print(f'[INFO] Loaded checkpoint from {model_path}')
    else:
        print(f'[WARN] model randomly initialized, are you sure?')

    # device
    model = model.half().to(device)

    rays_embeddings = model.prepare_default_rays(device)
    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    # 1. bg removal, also we can directly use the rgba image
    bg_remover = rembg.new_session()
    image_data = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
    mask = image_data[..., -1] > 0
    image = image_data.astype(np.float32) / 255.0
    # rgba to rgb white bg
    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4]) # [H, W, 3]
        image_data = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_data)
        pil_image.save(f"sample_mvdream_drag-rgba-white.png")

    # reshape [512,512,3] to [4, 256, 256,3]
    mv_image = image.reshape(2, 256, 2, 256, 3).transpose(0, 2, 1, 3, 4).reshape(4, 256, 256, 3)
    for i in range(4):
        mask = mv_image[i,...] > 0
        mv_image[i,...] = recenter(mv_image[i,...], mask, border_ratio=0.0)  

    # 2. generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256] (0~1)
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians, gaussians_list = model.forward_gaussians(input_image)
        
        # save gaussians
        model.gs.save_ply(gaussians, os.path.join(workspace_name, f'{stage}_3DGS.ply'))
        # for i in range(4):
        #     model.gs.save_ply(gaussians_list[i], os.path.join(workspace_name, f'{stage}_3DGS_{i}.ply'))

        # render 360 video 
        images = []
        depths = []
        elevation = 0

        if opt.fancy_video:
            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=1.5, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = min(azi / 360, 1)

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        else:
            azimuth = np.arange(0, 360, 3, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=1.5, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                out = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)
                image = out['image']
                image = (image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
                depth = out['depth']
                depth = depth.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy()
                images.append(image)
                depths.append(depth) # (1, 256, 256, 1)

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(os.path.join(workspace_name, f'{stage}_3DGS.mp4'), images, fps=30)
        
        eval_path = os.path.join(workspace_name, f'eval_{stage}')
        os.makedirs(eval_path, exist_ok=True)
        for i, image in enumerate(images):
            imageio.imwrite(os.path.join(eval_path, f'rgb_{i}.png'), image)

    # 2. project dragging points
    if isProject2D:
        # load src point from txt
        src_points = np.loadtxt(src_points_path)
        tgt_points = np.loadtxt(tgt_points_path)

        if src_points.ndim == 1:
            src_points = src_points.reshape(1, -1)
        if tgt_points.ndim == 1:
            tgt_points = tgt_points.reshape(1, -1)

        src_points = src_points[:,1:]
        ones_column = np.ones((tgt_points.shape[0], 1))
        src_points_with_ones = np.append(src_points, ones_column, axis=1)
        tgt_points = tgt_points[:,1:]
        tgt_points_with_ones = np.append(tgt_points, ones_column, axis=1)

        tracking_points_array = np.empty((0, src_points_with_ones.shape[1]), float)
        for i in range(src_points_with_ones.shape[0]):
            tracking_points_array = np.vstack((tracking_points_array, src_points_with_ones[i], tgt_points_with_ones[i]))
        tracking_points_tensor = torch.tensor(tracking_points_array, dtype=torch.float32, device=device)#torch.Size(4,4])

        txt_path = os.path.join(workspace_name, 'mv_drag_points_all.txt')
        azimuth = np.array([0, 90, 180, 270], dtype=np.int32)
        combined_image_list = []
        combined_masks_list = []
        img_proj_lines = []
        width = height = 256
        img_line = np.zeros((512, 512, 3), dtype=np.uint8)
        subimages = [mv_image[0], mv_image[1], mv_image[2], mv_image[3]]
        for iteration, azi in enumerate(tqdm.tqdm(azimuth)):
            cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=1.5, opengl=True)).unsqueeze(0).to(device)
            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]
            bg_color = torch.tensor([1/2, 1/2, 1/2], dtype=torch.float32, device=device)
            results = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), bg_color=bg_color, scale_modifier=1)
            image = results['image']
            mask = results['alpha']
            image = (image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
            mask = mask.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy()
            mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
            mask = mask.squeeze(0).squeeze(-1)
            pil_image = Image.fromarray(image.squeeze(0))
            pil_image.save(os.path.join(workspace_name, f'{stage}_view_{azi}.png'))
            combined_image_list.append(pil_image)

            pil_mask = Image.fromarray(mask).convert("L")  # Convert to grayscale
            pil_mask.save(os.path.join(workspace_name, f'{stage}_view_{azi}_mask.png'))
            combined_masks_list.append(pil_mask)

            tracking_points_2d_homogeneous = torch.matmul(cam_view[0].transpose(0, 1), tracking_points_tensor.t())     
            tracking_points_2d_homogeneous = torch.matmul(proj_matrix.transpose(0, 1), tracking_points_2d_homogeneous)
            tracking_points_depth = tracking_points_2d_homogeneous[2]
            tracking_points_2d_homogeneous = tracking_points_2d_homogeneous[:3]/(tracking_points_2d_homogeneous[3]+0.00000001) #ndc
            # ndc2Pix
            tracking_points_2d_homogeneous[0] = (((tracking_points_2d_homogeneous[0]+1.0)*256)-1)*0.5
            tracking_points_2d_homogeneous[1] = (((tracking_points_2d_homogeneous[1]+1.0)*256)-1)*0.5
            tracking_points_2d = tracking_points_2d_homogeneous.to(torch.int)
            tracking_points_2d = tracking_points_2d[:2]
            tracking_points_2d_cpu = tracking_points_2d.to('cpu')
            tracking_points_2d_cpu = tracking_points_2d_cpu.numpy()
            with open(txt_path, 'a') as f:
                np.savetxt(f, tracking_points_2d_cpu)
            # draw on input images
            img_proj_line = cv2.imread(os.path.join(workspace_name, f'{stage}_view_{azi}.png'))
            img_proj_lines.append(img_proj_line.copy())
            subimages[iteration] = draw_lines_on_subimage(img_proj_line, tracking_points_2d_cpu)
        
        img_line[0:256, 0:256] = subimages[0]
        img_line[0:256, 256:512] = subimages[1]
        img_line[256:512, 0:256] = subimages[2]
        img_line[256:512, 256:512] = subimages[3]
        img_line_save_path = os.path.join(workspace_name, 'mv_image_line.png')
        cv2.imwrite(img_line_save_path, img_line)

        # Create a 2x2 grid of the images
        top_row = np.hstack((img_proj_lines[0], img_proj_lines[1]))
        bottom_row = np.hstack((img_proj_lines[2], img_proj_lines[3]))
        combined_img = np.vstack((top_row, bottom_row))
        combined_img_save_path = os.path.join(workspace_name, f'mv_image.png')
        cv2.imwrite(combined_img_save_path, combined_img)

        combined_masks = Image.new('L', (width * 2, height * 2))
        combined_masks.paste(combined_masks_list[0], (0, 0))
        combined_masks.paste(combined_masks_list[1], (width, 0))
        combined_masks.paste(combined_masks_list[2], (0, height))
        combined_masks.paste(combined_masks_list[3], (width, height))
        combined_masks.save(os.path.join(workspace_name, 'mv_mask.png'))

