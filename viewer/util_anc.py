import numpy as np
from plyfile import PlyData
from utils.encodings import \
    STE_binary, STE_multistep, GridEncoder
from torch import nn
import torch
from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric, build_rotation)

class mix_3D2D_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list,
            log2_hashmap_size,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )

        self.output_dim = self.encoding_xyz.output_dim + \
                          self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]

        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        
        return out_i


class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int=32,
                 n_offsets: int=5,
                 n_features_per_level: int=2,
                 log2_hashmap_size: int=19,
                 log2_hashmap_size_2D: int=17,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                #  resolutions_list=(16, 23, 32, 46, 64, 92, 128, 184, 256, 368, 512, 736, 1024),
                #  resolutions_list_2D=(512, 1024, 2048, 4096),
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D


        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self._scaling = torch.empty(0)

        self.setup_functions()

        if use_2D:
            self.encoding_xyz = mix_3D2D_encoding(
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                resolutions_list_2D=resolutions_list_2D,
                log2_hashmap_size_2D=log2_hashmap_size_2D,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()
        else:
            self.encoding_xyz = GridEncoder(
                num_dim=3,
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()

        mlp_input_feat_dim = feat_dim

        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.mlp_grid = nn.Sequential(
            nn.Linear(self.encoding_xyz.output_dim, feat_dim*2),
            nn.ReLU(True),
            nn.Linear(feat_dim*2, (feat_dim+6+3*self.n_offsets)*2+1+1+1),
        ).cuda()

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_anchor(self):
        return self._anchor

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(False))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(False))
        scales = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._scaling = nn.Parameter(scales.requires_grad_(False))


    def load_mlp_checkpoints(self, path, load_hash_grid=True):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if load_hash_grid:
            self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
            self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])

        if 'x_bound_min' in checkpoint:
            self.x_bound_min = checkpoint['x_bound_min']
            self.x_bound_max = checkpoint['x_bound_max']


    def initial_for_P_frame(self, ntc_cfg):

        self.ntc1 = mix_3D2D_encoding(
                n_features=ntc_cfg['n_features_per_level'],
                resolutions_list=ntc_cfg["resolutions_list"],
                log2_hashmap_size=ntc_cfg["log2_hashmap_size"],
                resolutions_list_2D=ntc_cfg["resolutions_list_2D"],
                log2_hashmap_size_2D=ntc_cfg["log2_hashmap_size_2D"],
                ste_binary=self.ste_binary,
                ste_multistep=self.ste_multistep,
                add_noise=self.add_noise,
                Q=self.Q,
            ).cuda()

        self.ntc2 = mix_3D2D_encoding(
                n_features=ntc_cfg['n_features_per_level'],
                resolutions_list=ntc_cfg["resolutions_list"],
                log2_hashmap_size=ntc_cfg["log2_hashmap_size"],
                resolutions_list_2D=ntc_cfg["resolutions_list_2D"],
                log2_hashmap_size_2D=ntc_cfg["log2_hashmap_size_2D"],
                ste_binary=self.ste_binary,
                ste_multistep=self.ste_multistep,
                add_noise=self.add_noise,
                Q=self.Q,
            ).cuda()
                
        self.ntc_mlp1 = nn.Sequential(
            nn.Linear(self.ntc1.output_dim, self.feat_dim*2),
            nn.ReLU(True),
            nn.Linear(self.feat_dim*2, 3*self.n_offsets),
        ).cuda()

        self.ntc_mlp2 = nn.Sequential(
            nn.Linear(self.ntc2.output_dim, self.feat_dim*2),
            nn.ReLU(True),
            nn.Linear(self.feat_dim*2, self.feat_dim),
        ).cuda()

    def load_ntc_checkpoints(self, path, stage):
        checkpoint = torch.load(path)
        if stage == "stage1":
            self.ntc1.load_state_dict(checkpoint['ntc'])
            self.ntc_mlp1.load_state_dict(checkpoint['ntc_mlp'])
        elif stage == "stage2":
            self.ntc2.load_state_dict(checkpoint['ntc'])
            self.ntc_mlp2.load_state_dict(checkpoint['ntc_mlp'])                    

    def update_by_ntc(self):
        # x: [N, 3]
        x = self._anchor
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]

        mask = (x >= 0) & (x <= 1)
        mask = mask.all(dim=1)

        features = self.ntc1(x[mask])
        features = self.ntc_mlp1(features)
        d_offsets = torch.full((x.shape[0], 3*self.n_offsets), 0.0, dtype=torch.float32, device="cuda")
        d_offsets[mask] = features
        d_offsets = d_offsets.reshape(-1, self.n_offsets, 3)
        self._offset = nn.Parameter(self._offset + d_offsets)

        features = self.ntc2(x[mask])
        features = self.ntc_mlp2(features)
        d_feat = torch.full((x.shape[0], self.feat_dim), 0.0, dtype=torch.float32, device="cuda")
        d_feat[mask] = features
        self._anchor_feat = nn.Parameter(self._anchor_feat + d_feat)



