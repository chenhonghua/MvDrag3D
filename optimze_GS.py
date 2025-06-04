import os
import cv2
import time
import math
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
from torch import nn
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer_deform import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize
import imageio
import open3d as o3d
import copy
import lpips

import wandb
# from loss_utils import get_loss
import random

LPIPS = lpips.LPIPS(net='vgg').to(device='cuda')
# LPIPS.eval()
for param in LPIPS.parameters():
    param.requires_grad = False

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            pose1 = orbit_camera(self.opt.elevation, 0, self.opt.radius)
            pose2 = orbit_camera(self.opt.elevation, 90, self.opt.radius)
            pose3 = orbit_camera(self.opt.elevation, 180, self.opt.radius)
            pose4 = orbit_camera(self.opt.elevation, 270, self.opt.radius)
        else:
            pose1 = orbit_camera(self.opt.elevation, 0, self.opt.radius)
            pose2 = orbit_camera(self.opt.elevation, 90, self.opt.radius)
            pose3 = orbit_camera(self.opt.elevation, 180, self.opt.radius)
            pose4 = orbit_camera(self.opt.elevation, 270, self.opt.radius)
        self.fixed_cams = []
        self.fixed_cams.append(MiniCam(
            pose1,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,)
        )
        self.fixed_cams.append(MiniCam(
            pose2,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,)
        )
        self.fixed_cams.append(MiniCam(
            pose3,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,)
        )
        self.fixed_cams.append(MiniCam(
            pose4,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,)
        )        

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                self.guidance_sd.to(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream
                self.guidance_sd = ImageDream(self.device)
                print(f"[INFO] loaded ImageDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device) #torch.Size([1, 3, 512, 512])
            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                if self.opt.imagedream:
                    self.guidance_sd.get_image_text_embeds(self.input_img_torch[:,:, 0:512//2, 0:512//2], [self.prompt], [self.negative_prompt])
                else:
                    # pass
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch[:,:, 0:512//2, 0:512//2])
        
        # control the stage of deform or not or optimize gaussians
        self.updated = False

        # set known view
        self.subimages = [
            self.input_img_torch[:,:, 0:512//2, 0:512//2],
            self.input_img_torch[:,:, 0:512//2, 512//2:512],
            self.input_img_torch[:,:, 512//2:512, 0:512//2],
            self.input_img_torch[:,:, 512//2:512, 512//2:512]
        ] #0 ,1 // 2, 3
        self.submasks = [
            self.input_mask_torch[:,:, 0:512//2, 0:512//2],
            self.input_mask_torch[:,:, 0:512//2, 512//2:512],
            self.input_mask_torch[:,:, 512//2:512, 0:512//2],
            self.input_mask_torch[:,:, 512//2:512, 512//2:512]
        ] #0 ,1 // 2, 3

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0.0
            index = 0

            ### setting for performing deform or optimize gaussians
            if not self.opt.optimize_appearance and self.step <= 1500:
                gs_deform = self.renderer.gaussians.pos_deform()
                sh_deform = None
            elif not self.opt.optimize_appearance and self.step > 1500:
                gs_deform = None
                sh_deform = None
            elif self.opt.optimize_appearance:
                gs_deform = None
                sh_deform = None
            else:
                # set requires_grad to True
                # self.renderer.gaussians._xyz.requires_grad = False
                self.renderer.gaussians._features_dc.requires_grad = True
                self.renderer.gaussians._features_rest.requires_grad = True
                # update optimizer
                if not self.updated:
                    self.updated = True
                    print(f"[INFO] update optimizer...")
                    self.renderer.gaussians.update_optimizer(self.opt)
                    self.optimizer = self.renderer.gaussians.optimizer
                with torch.no_grad():
                    gs_deform = self.renderer.gaussians.pos_deform()
                    sh_deform = None
            
            if self.input_img_torch is not None:
                idx = [0,1,2,3]
                for i in range(4):
                    index = idx[i]
                    cur_cam = self.fixed_cams[index]
                    out = self.renderer.render(cur_cam, gs_deform=gs_deform, sh_deform=sh_deform)
                    image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    
                    # rgb loss
                    # rgb_loss = 10000 * F.mse_loss(image, self.subimages[i])
                    # loss = loss + rgb_loss
                    
                    # lpips loss
                    lpips_loss = 10000 * LPIPS(image, self.subimages[index]).mean()
                    loss = loss + lpips_loss

                    # mask loss
                    mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                    msk_loss = 100 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(mask, self.submasks[i])
                    loss = loss + msk_loss

            ## novel view (manual batch)
            render_resolution = 256 # if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)

            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                # radius = 0.0
                radius = random.uniform(-0.5, 0.5)

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
                out = self.renderer.render(cur_cam, bg_color=bg_color, gs_deform=gs_deform, sh_deform=sh_deform)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)

                # enable mvdream training
                if self.opt.mvdream or self.opt.imagedream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                        poses.append(pose_i)

                        cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                        bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
                        out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                        image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)
                    
            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            # guidance loss
            if self.enable_sd and self.step > 0:
                if  self.opt.imagedream:
                    index = random.randint(0, 3)
                    self.guidance_sd.get_image_text_embeds(self.subimages[index], [self.prompt], [self.negative_prompt])
                    sds_loss = self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio=step_ratio if self.opt.anneal_timestep else None)
                    loss = loss + sds_loss
                elif self.opt.mvdream:
                    sds_loss = self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio=step_ratio if self.opt.anneal_timestep else None)
                    loss = loss + sds_loss
                else:
                    sds_loss = self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio=step_ratio if self.opt.anneal_timestep else None)
                    loss = loss + sds_loss 

            if self.enable_zero123:
                sds_loss = self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.opt.anneal_timestep else None, default_elevation=self.opt.elevation)
                loss = loss + sds_loss
            
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if self.renderer.gaussians._features_dc.requires_grad and self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                print(f"[INFO] densify and prune...")
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()


        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True
        # print(f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}")        
        # wandb.log({"loss": loss, "rgb_loss": rgb_loss, "msk_loss": msk_loss, "sds_loss": sds_loss})
        
        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!
 
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()


    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 256

            import nvdiffrast.torch as dr
            # if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
            #     glctx = dr.RasterizeGLContext()
            # else:
            glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do a last prune
            if self.opt.optimize_appearance:
                self.renderer.gaussians.prune(min_opacity=0.05, extent=1, max_screen_size=1)
        
        # save videos
        with torch.no_grad():
            if not self.opt.optimize_appearance:
                ### perform deform
                gs_deform = self.renderer.gaussians.pos_deform()
                sh_deform = None
            else:
                gs_deform = None
                sh_deform = None

            azimuth = np.arange(0, 360, 3, dtype=np.int32)
            images = []
            depths = []
            for azi in tqdm.tqdm(azimuth):
                pose = orbit_camera(self.opt.elevation, azi, self.opt.radius)
                cur_cam = MiniCam(pose, 256, 256, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
                out = self.renderer.render(cur_cam, bg_color=bg_color, gs_deform=gs_deform, sh_deform=sh_deform)
                image = out["image"] #torch.Size([3, 256, 256])
                images.append((image.unsqueeze(0).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
                depth = out["depth"] #torch.Size([1, 256, 256])
                depth = depth.unsqueeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy()
                depths.append(depth) # (1, 256, 256, 1)
                
            # save image video
            images = np.concatenate(images, axis=0)
            video_path = os.path.join(self.opt.outdir, self.opt.save_path + '.mp4')
            imageio.mimwrite(video_path, images, fps=30)

            # save 120 images for gpt40 evaluation
            eval_path = os.path.join(self.opt.outdir, f'eval_{self.opt.stage}')
            os.makedirs(eval_path, exist_ok=True)
            for i, image in enumerate(images):
                imageio.imwrite(os.path.join(eval_path, f'rgb_{i}.png'), image)

            # save depth video
            depths = np.concatenate(depths, axis=0) # (180, 256, 256, 1)
            depths_normalize = (depths - depths.min()) / (depths.max() - depths.min())
            depths_normalize = (depths_normalize * 255).astype(np.uint8)
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_depth.mp4')
            imageio.mimwrite(path, depths_normalize, fps=30)

            # save GS
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_3DGS.ply')
            self.renderer.gaussians.save_ply(path, gs_deform=gs_deform)

            # save normal video by mesh
            print("running tsdf volume integration ...")
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length = 1/256,
                sdf_trunc=0.08,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
        
            for _ in range(1000):
                # render random view
                ver = np.random.randint(-90, 90)
                hor = np.random.randint(-180, 180)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius)
                cur_cam = MiniCam(pose, 256, 256, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
                out = self.renderer.render(cur_cam, bg_color=bg_color, gs_deform=gs_deform, sh_deform=sh_deform)
                image = out["image"] #torch.Size([3, 256, 256])
                depth = out["depth"] #torch.Size([1, 256, 256])

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.asarray(image.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                    o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                    depth_trunc = 1.5, convert_rgb_to_intensity=False,
                    depth_scale = 1.0
                )

                intrinsic=o3d.camera.PinholeCameraIntrinsic(width=cur_cam.image_width, 
                    height=cur_cam.image_height, 
                    cx = cur_cam.image_width/2,
                    cy = cur_cam.image_height/2,
                    fx = cur_cam.image_width / (2 * math.tan(cur_cam.FoVx / 2.)),
                    fy = cur_cam.image_height / (2 * math.tan(cur_cam.FoVy / 2.)))

                volume.integrate(rgbd, 
                    intrinsic=intrinsic, 
                    extrinsic=np.asarray((cur_cam.world_view_transform.T).cpu().numpy()))
            mesh = volume.extract_triangle_mesh()
            # write mesh
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.obj')
            o3d.io.write_triangle_mesh(path, mesh)

            mesh_0 = copy.deepcopy(mesh)
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

            # postprocessing
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)
            largest_cluster_idx = cluster_n_triangles.argmax()
            cluster_to_keep = min(len(cluster_n_triangles),1)
            n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]

            triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
            mesh_0.remove_triangles_by_mask(triangles_to_remove)
            mesh_0.remove_unreferenced_vertices()
            mesh_0.compute_vertex_normals()
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh_clean.obj')
            o3d.io.write_triangle_mesh(path, mesh_0)


    def save_mesh(self):
        # save videos
        with torch.no_grad():
            if self.opt.optimize_appearance:
                ### perform deform
                gs_deform = self.renderer.gaussians.pos_deform()
                sh_deform = None
            else:
                gs_deform = None
                sh_deform = None

            # mesh
            print("running tsdf volume integration ...")
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length = 1/256,
                sdf_trunc=0.08,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
        
            for _ in range(1000):
                # render random view
                ver = np.random.randint(-90, 90)
                hor = np.random.randint(-180, 180)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius)
                cur_cam = MiniCam(pose, 256, 256, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
                out = self.renderer.render(cur_cam, bg_color=bg_color, gs_deform=gs_deform, sh_deform=sh_deform)
                image = out["image"] #torch.Size([3, 256, 256])
                depth = out["depth"] #torch.Size([1, 256, 256])

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.asarray(image.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                    o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                    depth_trunc = 1.5, convert_rgb_to_intensity=False,
                    depth_scale = 1.0
                )

                intrinsic=o3d.camera.PinholeCameraIntrinsic(width=cur_cam.image_width, 
                    height=cur_cam.image_height, 
                    cx = cur_cam.image_width/2,
                    cy = cur_cam.image_height/2,
                    fx = cur_cam.image_width / (2 * math.tan(cur_cam.FoVx / 2.)),
                    fy = cur_cam.image_height / (2 * math.tan(cur_cam.FoVy / 2.)))

                volume.integrate(rgbd, 
                    intrinsic=intrinsic, 
                    extrinsic=np.asarray((cur_cam.world_view_transform.T).cpu().numpy()))
            mesh = volume.extract_triangle_mesh()
            # write mesh
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.obj')
            o3d.io.write_triangle_mesh(path, mesh)

            mesh_0 = copy.deepcopy(mesh)
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

            # postprocessing
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            cluster_area = np.asarray(cluster_area)
            largest_cluster_idx = cluster_n_triangles.argmax()
            cluster_to_keep = min(len(cluster_n_triangles),1)
            n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]

            triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
            mesh_0.remove_triangles_by_mask(triangles_to_remove)
            mesh_0.remove_unreferenced_vertices()
            mesh_0.compute_vertex_normals()
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh_clean.obj')
            o3d.io.write_triangle_mesh(path, mesh_0)

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters)
