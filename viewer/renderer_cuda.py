'''
Part of the code (CUDA and OpenGL memory transfer) is derived from https://github.com/jbaron34/torchwindow/tree/master
'''
from OpenGL import GL as gl
import OpenGL.GL.shaders as shaders
import util
import util_anc
import numpy as np
import torch
from renderer_ogl import GaussianRenderBase
from dataclasses import dataclass
from cuda import cudart as cu
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from einops import repeat


VERTEX_SHADER_SOURCE = """
#version 450

smooth out vec4 fragColor;
smooth out vec2 texcoords;

vec4 positions[3] = vec4[3](
    vec4(-1.0, 1.0, 0.0, 1.0),
    vec4(3.0, 1.0, 0.0, 1.0),
    vec4(-1.0, -3.0, 0.0, 1.0)
);

vec2 texpos[3] = vec2[3](
    vec2(0, 0),
    vec2(2, 0),
    vec2(0, 2)
);

void main() {
    gl_Position = positions[gl_VertexID];
    texcoords = texpos[gl_VertexID];
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330

smooth in vec2 texcoords;

out vec4 outputColour;

uniform sampler2D texSampler;

void main()
{
    outputColour = texture(texSampler, texcoords);
}
"""


@dataclass
class GaussianRasterizationSettingsStorage:
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool


class CUDARenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        self.raster_settings = {
            "image_height": int(h),
            "image_width": int(w),
            "tanfovx": 1,
            "tanfovy": 1,
            "bg": torch.Tensor([0., 0., 0]).float().cuda(),
            "scale_modifier": 1.,
            "viewmatrix": None,
            "projmatrix": None,
            "sh_degree": 1,  # ?
            "campos": None,
            "prefiltered": False,
            "debug": False
        }
        gl.glViewport(0, 0, w, h)
        self.program = util.compile_shaders(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
        # setup cuda
        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise RuntimeError(
                "OpenGL context may be running on integrated graphics"
            )
        
        self.vao = gl.glGenVertexArrays(1)
        self.tex = None
        self.NTC = None
        # the index of NTCs and additional_3dgs is the index of the current un-processed frame.
        self.NTCs = []
        self.additional_3dgs = []
        self.current_timestep=0
        self.set_gl_texture(h, w)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def update_gaussian_data(self, gaus: util_anc.GaussianModel):
        self.gaussians = gaus


    def sort_and_update(self, camera: util.Camera):
        pass
    
    def set_scale_modifier(self, modifier):
        self.raster_settings["scale_modifier"] = float(modifier)

    def set_render_mod(self, mod: int):
        pass
    
    def set_gl_texture(self, h, w):
        if self.tex is not None:
            gl.glDeleteTextures([self.tex])
        self.tex = gl.glGenTextures(1) 
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            w,
            h,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(
            self.tex,
            gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
        print(err)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to register opengl texture")
    
    def set_render_reso(self, w, h):
        self.raster_settings["image_height"] = int(h)
        self.raster_settings["image_width"] = int(w)
        gl.glViewport(0, 0, w, h)
        self.set_gl_texture(h, w)

    def fvv_reset(self):
        self.gaussians = self.init_gaussians.clone()
        self.current_timestep=0
        
    def update_camera_pose(self, camera: util.Camera):
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["viewmatrix"] = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings["campos"] = torch.tensor(camera.position).float().cuda()
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()

    def update_camera_intrin(self, camera: util.Camera):
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()
        hfovx, hfovy, focal = camera.get_htanfovxy_focal()
        self.raster_settings["tanfovx"] = hfovx
        self.raster_settings["tanfovy"] = hfovy

    def draw(self, camera: util.Camera, timestep: int = 0):
        raster_settings = GaussianRasterizationSettings(**self.raster_settings)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        xyz, color, opacity, scaling, rot = self.generate_neural_gaussians(camera, self.gaussians)
        with torch.no_grad():
            img, radii = rasterizer(
                means3D = xyz,
                means2D = None,
                shs = None,
                colors_precomp = color,
                opacities = opacity,
                scales = scaling,
                rotations = rot,
                cov3D_precomp = None
            )
        img = img.permute(1, 2, 0)
        img = torch.concat([img, torch.ones_like(img[..., :1])], dim=-1)
        img = img.contiguous()
        height, width = img.shape[:2]
        # transfer
        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to map graphics resource")
        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to get mapped array")
        
        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            img.data_ptr(),
            4 * 4 * width,
            4 * 4 * width,
            height,
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to copy from tensor to texture")
        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to unmap graphics resource")

        gl.glUseProgram(self.program)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)


    def generate_neural_gaussians(self, camera: util.Camera, pc : util_anc.GaussianModel, visible_mask=None):
        ## view frustum filtering for acceleration

        if visible_mask is None:
            visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

        anchor = pc.get_anchor[visible_mask]
        feat = pc._anchor_feat[visible_mask]
        grid_offsets = pc._offset[visible_mask]
        grid_scaling = pc.get_scaling[visible_mask]

        # feat = torch.tanh(feat)

        ob_view = anchor - torch.tensor(camera.position).float().cuda() # viewpoint_camera.camera_center
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        ob_view = ob_view / ob_dist

        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]

        neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N_visible_anchor, K]

        neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
        mask = (neural_opacity > 0.0)
        mask = mask.view(-1)  # [N_visible_anchor*K]

        # select opacity
        opacity = neural_opacity[mask]  # [N_opacity_pos_gaussian, 1]

        # get offset's color
        color = pc.get_color_mlp(cat_local_view)  # [N_visible_anchor, K*3]

        color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

        # get offset's cov
        scale_rot = pc.get_cov_mlp(cat_local_view)  # [N_visible_anchor, K*7]
        scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [N_visible_anchor*K, 7]

        offsets = grid_offsets.view([-1, 3])  # [N_visible_anchor*K, 3]

        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 6+3]
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [N_visible_anchor*K, 6+3]
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets],
                                    dim=-1)  # [N_visible_anchor*K, (6+3)+3+7+3]
        masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (6+3)+3+7+3]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

        # post-process cov
        scaling = scaling_repeat[:, 3:] * torch.sigmoid(
            scale_rot[:, :3])
        rot = pc.rotation_activation(scale_rot[:, 3:7])  # [N_opacity_pos_gaussian, 4]

        offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]
        xyz = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]

        return xyz, color, opacity, scaling, rot