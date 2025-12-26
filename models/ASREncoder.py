from models.Transfomer import Transformer
from models.SenseVoiceEncoderSmall import SenseVoiceEncoderSmall
import torch.nn as nn
from const import encoder_conf, adaptor_conf, input_size



class ASREncoder(nn.Module):
    def __init__(self):
        super().__init__()
          # audio_encoder
        self.sense_voice_encoder = SenseVoiceEncoderSmall(
            input_size=input_size, **encoder_conf)

        # adaptor
        self.adaptor = Transformer(**adaptor_conf)

    def forward(self, speech, speech_lengths):
        encoder_out, encoder_out_lens = self.sense_voice_encoder(speech, speech_lengths)
        encoder_out, encoder_out_lens = self.adaptor(encoder_out, encoder_out_lens)
        return encoder_out, encoder_out_lens

