import torch
import os
from models.FunASRNano import FunASRNano
from models.SenseVoiceEncoderSmall import SenseVoiceEncoderSmall
from const import encoder_conf, encoder_in_dim
model_path = "weights/Fun-ASR-Nano-2512"

if __name__ == "__main__":

    model, kwargs = FunASRNano.from_pretrained(model_path=model_path, device="cuda:0", disalbe_update=True)
    model.eval()
    
    audio_encoder = SenseVoiceEncoderSmall(input_size=encoder_in_dim, **encoder_conf)
    ckpt = torch.load(f"saved_models/audio_encoder.pt", map_location="cpu")
    audio_encoder.load_state_dict(ckpt)
    model.audio_encoder = audio_encoder
    
    
    audio_adaptor = Transformer(**adaptor_conf)
    ckpt = torch.load(f"saved_models/audio_adaptor.pt", map_location="cpu")
    audio_adaptor.load_state_dict(ckpt)
    model.audio_adaptor = audio_adaptor
    
    model.to("cuda:0")
    model.eval()
    wav_path = f"data/创建警单.wav"
    res = model.inference(wav_path=wav_path, **kwargs)
    text = res[0][0]["text"]
    print(text)
