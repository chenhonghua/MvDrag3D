import torch
import numpy as np
import kiui
import os
from mvdream_diffusers.mvdrag import mvdrag
from LGM.main_LGM import processLGM
from optimze_GS import GUI
import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32

def setup_workspace(opt):
    workspace_name = opt.workspace_name
    print(f'[INFO] Processing {workspace_name}')
    os.makedirs(workspace_name, exist_ok=True)
    return workspace_name

def run_lgm(opt, input_image, workspace_name):
    processLGM(
        input_image=input_image,
        model_path=opt.resume,
        workspace_name=workspace_name,
        isProject2D=opt.isProject2D,
        src_points_path=opt.src_points_path,
        tgt_points_path=opt.tgt_points_path,
        stage=opt.stage,
        device=device
    )

def save_mesh(gui, opt, save_path, workspace_name):
    opt.outdir = workspace_name
    opt.save_path = save_path
    gui.save_mesh()

def run_mvdrag(opt, workspace_name):
    opt.masks_path = f'{workspace_name}/mv_mask.png'
    opt.drag_points_path_all = f'{workspace_name}/mv_drag_points_all.txt'
    opt.drag_points_path_occ = f'{workspace_name}/mv_drag_points_occ.txt'
    mvdrag_image = mvdrag(opt=opt, device=device, dtype=dtype, workspace_name=workspace_name)
    grid = np.concatenate(
        [
            np.concatenate([mvdrag_image[0], mvdrag_image[2]], axis=0),
            np.concatenate([mvdrag_image[1], mvdrag_image[3]], axis=0),
        ],
        axis=1,
    )
    kiui.write_image(f'{workspace_name}/mv_drag_best_test.png', grid)

def mlp_deformation(gui, opt):
    gui.train(opt.iters)

def sds_appearance_optimization(gui, opt):
    gui.train(opt.iters)

def do(opt):
    start_time = time.time()
    workspace_name = setup_workspace(opt)
    opt.load = f'{workspace_name}/Initial_3DGS.ply'
    gui = GUI(opt)

    # 0. project 3D dragging points to the 4-view images
    # note that the dragging points are manually pre-picked by any 3D software (e.g. Meshlab), and saved in the workspace_name/keypoints
    opt.stage = "Initial"
    mv_input_image = kiui.read_image(opt.image_path, mode='uint8')
    opt.isProject2D = True
    run_lgm(opt, mv_input_image, workspace_name)
    opt.optimize_appearance = False
    # save_mesh(gui, opt, 'Initial_3DGS', workspace_name)
    print(f'[INFO] Running time for save mesh: {time.time() - start_time:.2f} seconds')
    start_time = time.time()

    # 1. mvdrag
    opt.stage = "Mvdrag"
    run_mvdrag(opt, workspace_name)
    print(f'[INFO] Running time for Drag: {time.time() - start_time:.2f} seconds')
    start_time = time.time()

    # # 2. use LGM to reconstruct the 3DGS from the mvdrag image
    opt.stage = "Mvdrag"
    opt.isProject2D = False
    mvdrag_image = kiui.read_image(f'{workspace_name}/mv_drag_best_test.png', mode='uint8') #note that you can offline remove the background of the mv_drag_best_test.png to get a better input for LGM
    run_lgm(opt, mvdrag_image, workspace_name)
    print(f'[INFO] Running time for 3D GS reconstruction: {time.time() - start_time:.2f} seconds')
    start_time = time.time()

    # 2.5 save LGM_3DGS
    # save_mesh(gui, opt, save_path='Mvdrag_3DGS', workspace_name=workspace_name)
    # print(f'[INFO] Running time for save mesh: {time.time() - start_time:.2f} seconds')
    # start_time = time.time()

    # 3. MLP deformation for misalignment optimization
    opt.optimize_appearance = False
    opt.stage = "Deformation"
    opt.iters = 4000
    opt.lambda_sd = 0.0
    opt.input = f'{workspace_name}/mv_drag_best_test.png'
    opt.load = f'{workspace_name}/Mvdrag_3DGS.ply'
    opt.outdir = workspace_name
    opt.save_path = opt.stage
    gui = GUI(opt)
    mlp_deformation(gui, opt)
    print(f'[INFO] Running time for MLP deformation: {time.time() - start_time:.2f} seconds')
    start_time = time.time()

    # 4. SDS appearance optimization
    opt.outdir = workspace_name
    opt.stage = "appearance"
    opt.optimize_appearance = True
    opt.lambda_sd = 1.0
    opt.input = f'{workspace_name}/mv_drag_best_test.png'
    opt.load = f'{workspace_name}/Deformation_3DGS.ply'
    opt.iters = 1000
    opt.save_path = opt.stage
    gui = GUI(opt)
    sds_appearance_optimization(gui, opt)
    print(f'[INFO] Running time for SDS: {time.time() - start_time:.2f} seconds')

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    do(opt)
