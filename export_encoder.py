

import torch
import os
from models.FunASRNano import FunASRNano
from models.SenseVoiceEncoderSmall import SenseVoiceEncoderSmall
from models.Transfomer import Transformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.ASREncoder import ASREncoder
from const import encoder_conf, encoder_in_dim, adaptor_conf
model_path = "weights/Fun-ASR-Nano-2512"
llm_path = "saved_models/llm/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":


    model, kwargs = FunASRNano.from_pretrained(model_path=model_path, device=device, disalbe_update=True)
    
    audio_encoder = model.audio_encoder
    audio_adaptor = model.audio_adaptor

    encoder = ASREncoder(audio_encoder, audio_adaptor)
    
    torch.save(encoder.state_dict(), "saved_models/encoder.pt")