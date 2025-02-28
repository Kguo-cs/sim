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


from src.smart.model.gail import GAIL
from src.smart.model.smart import SMART
from src.smart.modules.smart_decoder import SMARTDecoder

from src.smart.model.ego_gmm_smart import EgoGMMSMART
from src.smart.modules.ego_gmm_smart_decoder import EgoGMMSMARTDecoder
from src.smart.metrics import WOSACSubmission

class SMART_GAIL(GAIL, SMART):
    def __init__(self, model_config) -> None:
        SMART.__init__(self, model_config)  # Explicit call
        GAIL.__init__(self, model_config)  # Explicit call

        self.discriminator=SMARTDecoder(
            **model_config.decoder, n_token_agent=self.token_processor.n_token_agent,
            discrminator=True
        )



class EGO_GMM_GAIL(GAIL, EgoGMMSMART):
    def __init__(self, model_config) -> None:
        EgoGMMSMART.__init__(self, model_config)  # Explicit call
        GAIL.__init__(self, model_config)  # Explicit call

        self.discriminator = EgoGMMSMARTDecoder(**model_config.decoder)

    # def compute_dist(self):
    #     ego_pose_topk = torch.cat(
    #         [
    #             ego_next_poses[..., :2],
    #             ego_next_poses[..., [-1]].cos(),
    #             ego_next_poses[..., [-1]].sin(),
    #         ],
    #         dim=-1,
    #     )
    #     cov = (
    #         (self.gmm_cov * sampling_scheme.temp_cov)
    #         .repeat_interleave(2)[None, None, :]
    #         .expand(*ego_pose_topk.shape)
    #     )  # [n_batch, k, 4]
    #
    #     gmm = MixtureSameFamily(
    #         Categorical(logits=ego_next_logits), Independent(Normal(ego_next_poses, self.gmm_cov), 1)
    #     )
    #     ego_sample=ego_sample
    #
    #     prev_log_prob=gmm.log_prob(action)
    #
