import torch
import os
from models.FunASRNano import FunASRNano

model_path = "weights/Fun-ASR-Nano-2512"

if __name__ == "__main__":

    model, kwargs = FunASRNano.from_pretrained(model_path=model_path, device="cuda:0", disalbe_update=True)
    model.eval()
    
    audio_encoder = model.audio_encoder
    audio_adaptor = model.audio_adaptor
    llm = model.llm
    
    # 保存模型到本地
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存 audio_encoder
    torch.save(audio_encoder.state_dict(), os.path.join(save_dir, "audio_encoder.pt"))
    print(f"已保存 audio_encoder 到 {save_dir}/audio_encoder.pt")
    
    # 保存 audio_adaptor
    torch.save(audio_adaptor.state_dict(), os.path.join(save_dir, "audio_adaptor.pt"))
    print(f"已保存 audio_adaptor 到 {save_dir}/audio_adaptor.pt")
    
    # 保存 llm
    torch.save(llm.state_dict(), os.path.join(save_dir, "llm.pt"))
    print(f"已保存 llm 到 {save_dir}/llm.pt")
    
    wav_path = f"data/创建警单.wav"
    res = model.inference(wav_path=wav_path, **kwargs)
    text = res[0][0]["text"]
    print(text)
