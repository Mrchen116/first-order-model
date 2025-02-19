import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork


class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

        ######## zijian #########
        self.origin_image_grid = dict()
        self.fom_dict_weight = dict()

    def get_mix_origin_and_fom(self, deformation):
        device = deformation.device
        batch, h, w, _ = deformation.shape
        ori_deformation = torch.zeros(1, h, w, 2)
        fom_weight = torch.ones(1, h, w, 2)
        transition = h // 6
        for i in range(h):
            for j in range(w):
                dis = min([i, j, h - 1 - i, w - 1 - j]) + 1
                if dis > transition:
                    continue
                static = j * 2 / w - 1, i * 2 / h - 1
                ori_deformation[:, i, j, 0] = static[0] * (transition - dis) / transition
                ori_deformation[:, i, j, 1] = static[1] * (transition - dis) / transition
                fom_weight[:, i, j, 0] = dis / transition
                fom_weight[:, i, j, 1] = dis / transition

        # tot_h = h // 4
        # first_h = h - tot_h
        # for j in range(first_h, h):
        #     for i in range(w):
        #         static = i * 2 / w - 1, 1 - 2 / h * (h - j)
        #         ori_deformation[:, j, i, 0] = static[0] * (j - first_h) / tot_h
        #         ori_deformation[:, j, i, 1] = static[1] * (j - first_h) / tot_h
        #         fom_weight[:, j, i, 0] = (h - j) / tot_h
        #         fom_weight[:, j, i, 1] = (h - j) / tot_h
        self.origin_image_grid[(h, w)] = ori_deformation.to(device)
        self.fom_dict_weight[(h, w)] = fom_weight.to(device)

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        '''
        print('deformation', deformation.shape)
        print('input ', inp.shape)
        exit(0)
        '''
        if (h, w) not in self.origin_image_grid:
            self.get_mix_origin_and_fom(deformation)
        deformation = self.fom_dict_weight[(h, w)] * deformation + self.origin_image_grid[(h, w)]
        # tot_h = h // 3
        # first_h = h - tot_h
        # for j in range(first_h, h):
        #     for i in range(w):
        #         static = i * 2 / w - 1, 1 - 2 / h * (h - j)
        #         deformation[0, j, i, 0] = (static[0] * (j - first_h) + deformation[0, j, i, 0] * (h - j)) / tot_h
        #         deformation[0, j, i, 1] = (static[1] * (j - first_h) + deformation[0, j, i, 1] * (h - j)) / tot_h

        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                _, _, h, w = occlusion_map.shape
                # print("----------------- ", occlusion_map.shape, out.shape)
                fom_weight = self.fom_dict_weight[(h, w)][:,:,:,0].view(-1, h, w)
                occlusion_map = (1 - fom_weight) + fom_weight * occlusion_map
                out = out * occlusion_map

            output_dict["deformed"] = self.deform_input(source_image, deformation)

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict
