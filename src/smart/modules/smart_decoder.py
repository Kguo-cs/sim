"""Top-level SMART decoder.

This module combines:
    * map encoding;
    * agent-token policy decoding;
    * optional initial-state diffusion;
    * optional GAIL discriminator and value networks.

The public constructor, ``forward`` and ``inference`` signatures are compatible
with the previous implementation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor

from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.layers import MLPLayer
from src.smart.modules.agent_decoder import SMARTAgentDecoder
from src.smart.modules.map_decoder import SMARTMapDecoder


TensorDict = Dict[str, Tensor]


class SMARTDecoder(nn.Module):
    """Compose map, agent, initial-state, and discriminator decoders."""

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        pl2pl_radius: float,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        pt2pt_neighbor: int,
        pt2a_neighbor: int,
        a2a_neighbor: int,
        n_token_agent: int,
        dis_a2a_radius: float,
        dis_weight: float,
        dist_decay: float,
        reward_weight: float,
        reward_decay: float,
        token_processor=None,
        finetune: bool = False,
    ) -> None:
        super().__init__()

        self.token_processor = token_processor
        self.finetune = bool(finetune)


        self.gail = dis_a2a_radius > 0
        self.use_lcf = reward_weight != 0
        self.use_kl_penalty = False
        self.alpha = 0.1

        # External code reads these fields.
        self.pl2a_radius = pl2a_radius
        self.pt2a_neighbor = pt2a_neighbor

        self.map_encoder = self._make_map_encoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pt2pt_neighbor=pt2pt_neighbor,
            token_processor=token_processor,
        )

        self.agent_encoder = SMARTAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
            pt2a_neighbor=pt2a_neighbor,
            a2a_neighbor=a2a_neighbor,
            token_processor=token_processor,
            alpha=self.alpha,
            dis_weight=dis_weight,
            dist_decay=dist_decay,
            reward_weight=reward_weight,
            reward_decay=reward_decay,
            use_gail=self.gail,
        )

        # Define optional attributes in every configuration.
        self.init_decoder: Optional[InitDiffusion] = None
        self.init_map_encoder: Optional[SMARTMapDecoder] = None
        self.sep_map = True

        self.discriminator: Optional[SMARTAgentDecoder] = None
        self.value_network: Optional[MLPLayer] = None
        self.nei_value_network: Optional[MLPLayer] = None
        self.init_value_network: Optional[MLPLayer] = None

        if self.token_processor.pred_init:
            self._build_initial_decoder(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_freq_bands=num_freq_bands,
                pl2pl_radius=pl2pl_radius,
                dropout=dropout,
                head_dim=head_dim,
                pt2pt_neighbor=pt2pt_neighbor,
            )

        if self.gail:
            self._build_gail_modules(
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                time_span=time_span,
                pl2a_radius=pl2a_radius,
                dis_a2a_radius=dis_a2a_radius,
                num_freq_bands=num_freq_bands,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                hist_drop_prob=hist_drop_prob,
                pt2a_neighbor=pt2a_neighbor,
                a2a_neighbor=a2a_neighbor,
                dis_weight=dis_weight,
                dist_decay=dist_decay,
                reward_weight=reward_weight,
                reward_decay=reward_decay,
            )

    @staticmethod
    def _make_map_encoder(
        *,
        hidden_dim: int,
        pl2pl_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        pt2pt_neighbor: int,
        token_processor,
    ) -> SMARTMapDecoder:
        return SMARTMapDecoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pt2pt_neighbor=pt2pt_neighbor,
            token_processor=token_processor,
        )

    def _build_initial_decoder(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_freq_bands: int,
        pl2pl_radius: float,
        dropout: float,
        head_dim: int,
        pt2pt_neighbor: int,
    ) -> None:
        self.init_decoder = InitDiffusion(
            hidden_dim,
            num_heads,
            num_freq_bands,
            self.token_processor,
            self.gail,
        )

        if not self.sep_map:
            return

        self.init_map_encoder = self._make_map_encoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pt2pt_neighbor=pt2pt_neighbor,
            token_processor=self.token_processor,
        )

    def _build_gail_modules(
        self,
        *,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        dis_a2a_radius: float,
        num_freq_bands: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        pt2a_neighbor: int,
        a2a_neighbor: int,
        dis_weight: float,
        dist_decay: float,
        reward_weight: float,
        reward_decay: float,
    ) -> None:
        self.discriminator = SMARTAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=10,
            pl2a_radius=pl2a_radius,
            a2a_radius=dis_a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=1,
            pt2a_neighbor=pt2a_neighbor,
            a2a_neighbor=a2a_neighbor,
            token_processor=self.token_processor,
            alpha=self.alpha,
            dis_weight=dis_weight,
            dist_decay=dist_decay,
            reward_weight=reward_weight,
            reward_decay=reward_decay,
            discriminator=True,
        )

        self.value_network = MLPLayer(hidden_dim, hidden_dim * 2, 1)


        if self.token_processor.learn_init:
            self.init_value_network = MLPLayer(
                self.init_decoder.G1.hidden_dim,
                hidden_dim * 2,
                1,
            )

        # Compatibility with existing training code.
        self.agent_encoder.interative_decoder.gail = True

    def _get_map_feature(
        self,
        tokenized_map: TensorDict,
        tokenized_agent: TensorDict,
    ):
        map_feature = tokenized_agent.get("map_feature")
        if map_feature is None:
            map_feature = self.map_encoder(tokenized_map)
            tokenized_agent["map_feature"] = map_feature
        return map_feature

    def _prepare_initial_map_feature(
        self,
        tokenized_map: TensorDict,
        tokenized_agent: TensorDict,
        map_feature,
    ):
        if not self.token_processor.pred_init:
            return None

        cached = tokenized_agent.get("initial_map_feature")
        if cached is not None:
            return cached

        if self.sep_map:
            if self.init_map_encoder is None:
                raise RuntimeError(
                    "sep_map=True but init_map_encoder is missing."
                )
            initial_map_feature = self.init_map_encoder(
                tokenized_map,
                tokenized_agent=tokenized_agent,
            )
            tokenized_agent["initial_map_feature"] = initial_map_feature

        else:
            tokenized_agent["map_feature"] = map_feature

    def forward(
        self,
        tokenized_map: TensorDict,
        tokenized_agent: TensorDict,
    ) -> TensorDict:
        if  self.sep_map and self.token_processor.learn_init and not self.gail:
            map_feature = None
        else:
            map_feature = self._get_map_feature(
                tokenized_map,
                tokenized_agent,
            )

        if self.token_processor.learn_init and not self.gail:
            prediction: TensorDict = {}
        else:
            prediction = self.agent_encoder(
                tokenized_agent,
                map_feature,
            )

        if self.token_processor.learn_init and not self.gail:
            self._prepare_initial_map_feature(
                tokenized_map,
                tokenized_agent,
                map_feature,
            )
            prediction["initial_logit"] = self.init_decoder(
                tokenized_agent
            )

        return prediction

    def inference(
        self,
        tokenized_agent: TensorDict,
        n_step_future_10hz: Optional[int] = None,
    ) -> TensorDict:

        return self.agent_encoder.inference(
            self.init_decoder,
            tokenized_agent,
            tokenized_agent["map_feature"],
            n_step_future_10hz=n_step_future_10hz,
        )
