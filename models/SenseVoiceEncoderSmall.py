import torch
import torch.nn as nn
from typing import Optional
from funasr.models.sense_voice.model import SinusoidalPositionEncoder, PositionwiseFeedForward, MultiHeadedAttentionSANM, EncoderLayerSANM, LayerNorm, sequence_mask

class SenseVoiceEncoderSmall(nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    SCAMA: Streaming chunk-aware multihead attention for online end-to-end speech recognition
    https://arxiv.org/abs/2006.01713
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        tp_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        selfattention_layer_type: str = "sanm",
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size

        self.embed = SinusoidalPositionEncoder()

        self.normalize_before = normalize_before

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
        )

        encoder_selfattn_layer = MultiHeadedAttentionSANM
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )

        self.encoders0 = nn.ModuleList(
            [
                EncoderLayerSANM(
                    input_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args0),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(1)
            ]
        )
        self.encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(num_blocks - 1)
            ]
        )

        self.tp_encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                )
                for i in range(tp_blocks)
            ]
        )

        self.after_norm = LayerNorm(output_size)

        self.tp_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ):
        """Embed positions in tensor."""
        maxlen = xs_pad.shape[1]
        masks = sequence_mask(ilens, maxlen=maxlen, device=ilens.device)[:, None, :]

        xs_pad *= self.output_size() ** 0.5

        xs_pad = self.embed(xs_pad)

        # forward encoder1
        for layer_idx, encoder_layer in enumerate(self.encoders0):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        for layer_idx, encoder_layer in enumerate(self.encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.after_norm(xs_pad)

        # forward encoder2
        olens = masks.squeeze(1).sum(1).int()

        for layer_idx, encoder_layer in enumerate(self.tp_encoders):
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.tp_norm(xs_pad)
        return xs_pad, olens