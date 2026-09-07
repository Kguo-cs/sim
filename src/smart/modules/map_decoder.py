# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from .edge_encoder import EdgeEncoder
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    angle_between_2d_vectors,
    weight_init
)

from src.smart.layers import MLPLayer

class SMARTMapDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        pl2pl_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        pt2pt_neighbor:int,
        token_processor
    ) -> None:
        super(SMARTMapDecoder, self).__init__()
        self.pl2pl_radius = pl2pl_radius
        self.num_layers = num_layers
        self.pt2pt_neighbor=pt2pt_neighbor

        self.token_processor=token_processor

        self.type_pt_emb = nn.Embedding(10, hidden_dim)
        self.polygon_type_emb = nn.Embedding(4, hidden_dim)
        self.light_pl_emb = nn.Embedding(5, hidden_dim)

        self.token_emb = MLPEmbedding(input_dim=22, hidden_dim=hidden_dim)

        self.edge_encoder = EdgeEncoder(hidden_dim, num_freq_bands, use_pl2a=True)

        self.pt2pt_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.apply(weight_init)

    def forward(self, tokenized_map: Dict,tokenized_agent=None):

        map_type=tokenized_map["type"].long()

        batch = tokenized_map["batch"]
        pos_pt = tokenized_map["position"]
        orient_pt = tokenized_map["orientation"]
        token_idx=tokenized_map["token_idx"].long()
        light_type=tokenized_map["light_type"].long()

        if tokenized_agent is None:
            mask = (map_type == 4) | (map_type == 5)
        else:
            gt_initial_pos = tokenized_agent["initial_pos"]
            ego_mask = tokenized_agent["ego_mask"]

            ego_position = gt_initial_pos[ego_mask].reshape(-1,batch.max().item()+1,2)

            dist=torch.norm(ego_position[:,batch]-pos_pt[None],dim=-1).amin(0)

            dist_mask=dist<(self.token_processor.init_map_range+self.pl2pl_radius)

            batch=batch[dist_mask]
            pos_pt=pos_pt[dist_mask]
            orient_pt=orient_pt[dist_mask]
            token_idx=token_idx[dist_mask]
            light_type=light_type[dist_mask]
            map_type=map_type[dist_mask]
            mask = (dist[dist_mask]<self.token_processor.init_map_range) & ((map_type == 4) | (map_type == 5))#& (map_type<4)#

            # valid_idx = torch.where(mask)[0]
            # mask[valid_idx[1::6]] = False
            # mask[valid_idx[2::6]] = False
            # mask[valid_idx[3::6]] = False
            # mask[valid_idx[4::6]] = False
            # mask[valid_idx[5::6]] = False

        pt_token_emb_src = self.token_emb(self.token_processor.map_token_traj_src)
        x_pt = pt_token_emb_src[token_idx]

        pl_type_mapping= torch.tensor([0,0,0,0,1,1,2,2,2,3,3,3]).to(device=pos_pt.device, dtype=torch.long)
        pl_type=pl_type_mapping[map_type]

        x_pt_categorical_embs = [
            self.type_pt_emb(map_type),
            self.polygon_type_emb(pl_type),
            self.light_pl_emb(light_type),
        ]

        x_pt = x_pt + torch.stack(x_pt_categorical_embs).sum(dim=0)

        if self.num_layers>1:
            head_vector = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)

            edge_index_pt2pt, r_pt2pt = self.edge_encoder.build_map2map_edge(
                pos_pt,  # [n_pl, 2]
                orient_pt,  # [n_pl]
                pos_pt,  # [n_agent, n_step, 2]
                orient_pt,  # [n_agent, n_step]
                head_vector,  # [n_agent, n_step, 2]
                batch,  # [n_agent*n_step]
                batch,  # [n_pl*n_step]
                self.pl2pl_radius,
                self.pt2pt_neighbor,
            )

            for i in range(self.num_layers):
                x_pt ,_= self.pt2pt_layers[i]((x_pt, x_pt), r_pt2pt, edge_index_pt2pt)


            x_pt=x_pt[mask]
            pos_pt=pos_pt[mask]
            orient_pt=orient_pt[mask]
            batch=batch[mask]

        elif self.num_layers>0:
            edge_pt=x_pt[mask]#[::2]
            pos_edge=pos_pt[mask]#[::2]
            orient_edge=orient_pt[mask]#[::2]
            batch_edge=batch[mask]#[::2].contiguous()

            head_vector_edge = torch.stack([orient_edge.cos(), orient_edge.sin()], dim=-1)

            edge_index_pt2pt, r_pt2pt = self.edge_encoder.build_map2map_edge(
                                                            pos_pt,  # [n_pl, 2]
                                                            orient_pt,  # [n_pl]
                                                            pos_edge,  # [n_agent, n_step, 2]
                                                            orient_edge,  # [n_agent, n_step]
                                                            head_vector_edge,  # [n_agent, n_step, 2]
                                                            batch_edge,  # [n_agent*n_step]
                                                            batch,  # [n_pl*n_step]
                                                            self.pl2pl_radius,
                                                            self.pt2pt_neighbor,
                                                        )

            edge_pt = self.pt2pt_layers[0]((x_pt, edge_pt), r_pt2pt, edge_index_pt2pt)

            x_pt=edge_pt
            pos_pt=pos_edge
            orient_pt=orient_edge
            batch=batch_edge

        if tokenized_agent is not None:
            ego_mask = tokenized_agent["ego_mask"]
            ego_position = tokenized_agent["initial_pos"][ego_mask]
            ego_heading = tokenized_agent["initial_heading"][ego_mask]

            pos_pt, orient_pt = transform_to_local(pos_pt,  # [:,None],
                                                   orient_pt,  # [:,None],
                                                   ego_position[batch],
                                                   ego_heading[batch],
                                                   )

        output={
            "pt_token": x_pt,
            "position": pos_pt,
            "orientation": orient_pt,
            "batch": batch,
        }


        return output

        #tensor([0.010, 0.488, 0.020, 0.034, 0.145, 0.025, 0.063, 0.071, 0.017, 0.127],
        # polyline_type = {
        #     # for lane
        #     "TYPE_FREEWAY": 0,
        #     "TYPE_SURFACE_STREET": 1,
        #     "TYPE_STOP_SIGN": 2,
        #     "TYPE_BIKE_LANE": 3,
        #     # for roadedge
        #     "TYPE_ROAD_EDGE_BOUNDARY": 4,
        #     "TYPE_ROAD_EDGE_MEDIAN": 5,
        #     # for roadline
        #     "BROKEN": 6,
        #     "SOLID_SINGLE": 7,
        #     "DOUBLE": 8,
        #     # for crosswalk, speed bump and drive way
        #     "TYPE_CROSSWALK": 9,
        # }


        # lengths = torch.bincount(batch).tolist()
        #
        # padded_pt_feature = padding(x_pt, lengths)
        #
        # feature_mask=(padded_pt_feature == 0).all(-1)
        #
        # map_mask = feature_mask[:, None]
        #
        # sinusoidal_pos=self.rotary_embedding(pos_pt, orient_pt)
        #
        # map_sinusoidal = padding(sinusoidal_pos, lengths)
        #
        # padd_pos=padding(pos_pt, lengths)
        # padd_heading=padding(orient_pt, lengths)
        #
        # # if not self.gnn:
        # #
        # #     pt2pt_mask = feature_mask[:, :, None] | feature_mask[:, None]
        # #
        # #     pt2pt_mask=nearest_mask(padd_pos,10,self.pl2pl_radius,pt2pt_mask)
        # #
        # #     padded_pt_feature = self.pt2pt_roformer(padded_pt_feature, pt2pt_mask[:,None], map_sinusoidal)
        # #
        # #     x_pt = padded_pt_feature[~feature_mask]
        #
        # return {
        #     "pt_token": x_pt,
        #     "padded_pt": padded_pt_feature,
        #     "position": pos_pt,
        #     "orientation": orient_pt,
        #     "padd_pos": padd_pos,
        #     "padd_heading": padd_heading,
        #
        #     # "centering_pos":centering_pos,
        #   #  "centering_heading":centering_heading,
        #     "rotary_embedding":self.rotary_embedding,
        #     "batch": batch,
        #     "map_mask": map_mask,
        #     "map_sinusoidal": map_sinusoidal
        # }

        #pos_pt1=pos_pt+torch.tensor(np.array([[10,100]])).to(device=x_pt.device)

        #orient_pt1=orient_pt+torch.pi*2


        # sinusoidal_pos = general_rope(pos_pt1, self.head_dim, orient_pt1)

        # map_sinusoidal1 = self.padding(sinusoidal_pos, lengths)

        # padd_pos=self.padding(pos_pt1, lengths)

        # pt2pt_dist=torch.linalg.norm(padd_pos[:,None]-padd_pos[:,:,None],dim=-1)

        # pt2pt_mask = map_mask | (pt2pt_dist>self.pl2pl_radius) | (pt2pt_dist==0)

        # x_pt1 = self.pt2pt_roformer(padded_pt_feature, pt2pt_mask[:,None], map_sinusoidal1)


