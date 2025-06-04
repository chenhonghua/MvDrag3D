<div align="center">

<h1>
MvDrag3D: Drag-based Creative 3D Editing via Multi-view Generation-Reconstruction Priors
</h1>

<a href="https://chenhonghua.github.io/clay.github.io/">Honghua Chen, </a></span>
<span class="author-block">
<a href="https://nirvanalan.github.io/">Yushi Lan, </a></span>
<span class="author-block">
<a href="https://cyw-3d.github.io/">Yongwei Chen, </a></span>
<span class="author-block">
<a href="https://zhouyifan.net/">Yifan Zhou, </a></span>
<span class="author-block">
<a href="https://xingangpan.github.io/">Xingang Pan</a></span>


<span class="author-block">S-Lab, Nanyang Technological University, Singapore</span>

</div>



<div id="carousel" class="results-carousel">
<img src="images/teaser.png" width="100%" alt="overview_image">
<div class="content has-text-justified">
    <p>
    <b>MVDrag3D</b> provides a precise, generative, and flexible solution for 3D drag-based editing, 
    supporting more versatile editing effects across various object categories and 3D representations.
    </p>
</div>
</div>

<h2 class="title is-3">Abstract</h2>
<div class="content has-text-justified">
    <p>
    Drag-based editing has become popular in 2D content creation, driven by the capabilities of image generative models. However, extending this technique to 3D remains a challenge. 
    Existing 3D drag-based editing methods, whether employing explicit spatial transformations or relying on implicit latent optimization within limited-capacity 3D generative models, 
    fall short in handling significant topology changes or generating new textures across diverse object categories. 
    To overcome these limitations, we introduce MVDrag3D, a novel framework for more flexible and creative drag-based 3D editing that leverages multi-view generation and reconstruction priors.
    At the core of our approach is the usage of a multi-view diffusion model as a strong generative prior to perform consistent drag editing over multiple rendered views, which is followed by a reconstruction model that reconstructs 3D Gaussians of the edited object.
    While the initial 3D Gaussians may suffer from misalignment between different views, we address this via view-specific deformation networks that adjust the position of Gaussians to be well aligned.
    In addition, we propose a multi-view score function that distills generative priors from multiple views to further enhance the view consistency and visual quality. Extensive experiments demonstrate that MVDrag3D provides a precise, generative, and flexible solution for 3D drag-based editing, supporting more versatile editing effects across various object categories and 3D representations.
    </p>
</div>

<h2 class="title is-3">Pipeline</h2>
<img style='height: auto; width: 100%; object-fit: contain' src="images/overview.png">
<div class="content has-text-justified">
<p>
    <strong>The overall architecture of MVDrag3D.</strong>
    Given a 3D model and multiple pairs of 3D dragging points, we first render the model into four orthogonal views, each with corresponding projected dragging points. 
    Then, to ensure consistent dragging across these views, we define a multi-view guidance energy within a multi-view diffusion model. 
    The resulting dragged images are used to regress an initial set of 3D Gaussians. 
    Our method further employs a two-stage optimization process: first, a deformation network adjusts the positions of the Gaussians for improved geometric alignment, 
    followed by image-conditioned multi-view score distillation to enhance the visual quality of the final output.
</p>
</div>


## User Instructions

### 1. Clone the repository
```bash
git clone https://github.com/chenhonghua/MvDrag3D.git
cd MvDrag3D
```

### 2. Install dependencies
Install the required dependencies. For example:
```bash
pip install -r requirements.txt
```
Or, if you use conda:
```bash
conda env create -f environment.yml
conda activate your_env_name
```

### 3. Prepare your data
- Place your images, keypoints, and other data in the appropriate directories (e.g., `MvDrag3D/dragonGaussian/viking_axe2/`).

- Prepare the `src_points_path` and `tgt_points_path` files by following the provided example formats.

- Keypoints should be manually selected using tools like **MeshLab** or **CloudCompare**.  
- Source 3D keypoints should be saved in:
  ```
  srcs_single_keypoints.txt
  ```
- Corresponding target keypoints should be saved in:
  ```
  user_single_keypoints.txt
  ```

- For dragging on a 3D Gaussian Splatting (3DGS) scene:
- Use **LGM** to generate the initial 3DGS point cloud:
  ```
  Initial_3DGS.ply
  ```
- The associated input image should be:
  ```
  test_mvdream_0.png
  ```

- The file `mv_drag_points_occ.txt` contains the projected dragging points from four different viewpoints. Note that: If a point is occluded in any view, its projected coordinate can be set to zero—either automatically using depth information or manually.



### 4. Run the main program
You can use the provided bash script:
```bash
bash MvDrag3D/bash_test.sh
```
Or run the Python script directly:
```bash
CUDA_VISIBLE_DEVICES=0 python main_me.py --config configs/configs.yaml ...
```
(Refer to `bash_test.sh` for parameter examples.)

### 5. View results
- The results will be saved in the directory specified by the `workspace_name` parameter.
- Initial_3DGS:
  ```
  Initial_3DGS.ply
  ```
- After performing multi-view dragging, the editing result is:
  ```
  mv_drag_best_test.png
  ```
- Based on the dragged image, a coarse 3DGS is reconstructed using LGM:
  ```
  Mvdrag_3DGS.ply
  ```
- Finally, a two-stage optimization is applied: **Deformation Optimization** and **Appearance Optimization**. We otain:
  ```
  Deformation_3DGS.ply and appearance_3DGS.ply.
  ```

---
For more details on parameters, optional features, or troubleshooting, please refer to other documentation in the repository or open an issue.


---
## Acknowledgements

This project is built upon the great work of the following code bases:

- [mvdream-diffusers](https://github.com/lizhiqi49/mvdream-diffusers)
- [LGM](https://github.com/3DTopia/LGM)
- [Dreamgaussian](https://github.com/dreamgaussian/dreamgaussian)

We sincerely thank the authors for their contributions to the community.


<section class="section" id="BibTeX">
<div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>
@article{chen2024mvdrag3d,
  title={MvDrag3D: Drag-based Creative 3D Editing via Multi-view Generation-Reconstruction Priors},
  author={Chen, Honghua and Lan, Yushi and Chen, Yongwei and Zhou, Yifan and Pan, Xingang},
  journal={arXiv preprint arXiv:2410.16272},
  year={2024}
}</code></pre>
</div>
</section>