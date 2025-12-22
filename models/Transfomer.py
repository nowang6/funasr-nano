import torch
import torch.nn as nn
import torch.nn.functional as F
from funasr.models.transformer.encoder import EncoderLayer
from funasr.models.transformer.attention import MultiHeadedAttention
from funasr.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from funasr.models.transformer.utils.nets_utils import make_pad_mask


class Transformer(nn.Module):
    def __init__(
        self, downsample_rate=2, encoder_dim=1280, llm_dim=4096, ffn_dim: int = 2048, **kwargs
    ):
        super().__init__()
        self.k = downsample_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, self.llm_dim)
        from funasr.models.transformer.encoder import EncoderLayer
        from funasr.models.transformer.attention import MultiHeadedAttention
        from funasr.models.transformer.positionwise_feed_forward import PositionwiseFeedForward

        self.blocks = None
        if kwargs.get("n_layer", 2) > 0:
            self.blocks = nn.ModuleList(
                [
                    EncoderLayer(
                        llm_dim,
                        MultiHeadedAttention(
                            kwargs.get("attention_heads", 8),
                            llm_dim,
                            kwargs.get("attention_dropout_rate", 0.0),
                        ),
                        PositionwiseFeedForward(
                            llm_dim,
                            llm_dim // 4,
                            kwargs.get("dropout_rate", 0.0),
                        ),
                        kwargs.get("dropout_rate", 0.0),
                    )
                    for i in range(kwargs.get("n_layer", 2))
                ]
            )

    def forward(self, x, ilens=None):

        batch_size, seq_len, dim = x.size()
        # num_frames_to_discard = seq_len % self.k
        chunk_num = (seq_len - 1) // self.k + 1
        pad_num = chunk_num * self.k - seq_len
        x = F.pad(x, (0, 0, 0, pad_num, 0, 0), value=0.0)
        # if num_frames_to_discard > 0:
        #     x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, chunk_num, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        olens = None
        olens = (ilens - 1) // self.k + 1
        masks = (~make_pad_mask(olens)[:, None, :]).to(x.device)

        if self.blocks is not None:
            for layer, block in enumerate(self.blocks):
                x, masks = block(x, masks)
        return x, olens
