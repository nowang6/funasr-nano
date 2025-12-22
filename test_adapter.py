import torch
import os
from models.Transfomer import Transformer
from const import adaptor_conf

save_dir = "saved_tensors"

if __name__ == "__main__":

    audio_adaptor = Transformer(**adaptor_conf)
    ckpt = torch.load(f"saved_models/audio_adaptor.pt", map_location="cpu")
    audio_adaptor.load_state_dict(ckpt)
    audio_adaptor.eval()
    
    # 从文件加载输入和输出
    # adapter 的输入是 encoder 的输出
    # adapter 的输出是 audio_adaptor_out
    print("Loading tensors from files...")
    encoder_out_input = torch.load(os.path.join(save_dir, "encoder_out.pt"))  # adapter 的输入
    encoder_out_lens_input = torch.load(os.path.join(save_dir, "encoder_out_lens.pt"))  # adapter 的输入长度
    adaptor_out_original = torch.load(os.path.join(save_dir, "audio_adaptor_out.pt"))  # adapter 的原始输出
    
    print(f"Input shapes (adapter input = encoder output):")
    print(f"  encoder_out: {encoder_out_input.shape}")
    print(f"  encoder_out_lens: {encoder_out_lens_input.shape}")
    print(f"Original output shapes (adapter output):")
    print(f"  adaptor_out: {adaptor_out_original.shape}")
    
    # 将输入移到正确的设备
    device = next(audio_adaptor.parameters()).device
    encoder_out_input = encoder_out_input.to(device)
    encoder_out_lens_input = encoder_out_lens_input.to(device)
    
    # 使用 audio_adaptor 进行推理
    print("\nRunning inference with audio_adaptor...")
    with torch.no_grad():
        adaptor_out_new, adaptor_out_lens_new = audio_adaptor(
            encoder_out_input, encoder_out_lens_input
        )
    
    print(f"New output shapes:")
    print(f"  adaptor_out: {adaptor_out_new.shape}")
    print(f"  adaptor_out_lens: {adaptor_out_lens_new.shape}")
    
    # 将原始输出也移到相同设备以便比较，并确保数据类型一致
    adaptor_out_original = adaptor_out_original.to(device).to(adaptor_out_new.dtype)
    
    # 简单比较结果差异
    print("\n" + "="*60)
    print("Comparing results...")
    print("="*60)
    
    # 比较 adaptor_out
    if adaptor_out_original.shape == adaptor_out_new.shape:
        mae = torch.mean(torch.abs(adaptor_out_original - adaptor_out_new)).item()
        is_match = torch.allclose(adaptor_out_original, adaptor_out_new, rtol=1e-5, atol=1e-8)
        print(f"\nAdaptor Output:")
        print(f"  MAE: {mae:.6e}")
        print(f"  Match: {'✓ Yes' if is_match else '✗ No'}")
    else:
        print(f"\nAdaptor Output: Shape mismatch!")
        print(f"  Original: {adaptor_out_original.shape}, New: {adaptor_out_new.shape}")
    
    # 比较 adaptor_out_lens（如果有保存的话）
    print(f"\nAdaptor Output Lengths: {adaptor_out_lens_new}")
    
    print("="*60)