from models.FunASRNano import FunASRNano
import torch
import os

model_path = "weights/Fun-ASR-Nano-2512"
save_dir = "saved_tensors"

if __name__ == "__main__":

    model, kwargs = FunASRNano.from_pretrained(model_path=model_path, device="cuda:0", disalbe_update=True)
    model.eval()
    
    audio_encoder = model.audio_encoder
    audio_encoder.eval()
    
    # 从文件加载输入和输出
    print("Loading tensors from files...")
    speech = torch.load(os.path.join(save_dir, "speech.pt"))
    speech_lengths = torch.load(os.path.join(save_dir, "speech_lengths.pt"))
    encoder_out_original = torch.load(os.path.join(save_dir, "encoder_out.pt"))
    encoder_out_lens_original = torch.load(os.path.join(save_dir, "encoder_out_lens.pt"))
    
    print(f"Input shapes:")
    print(f"  speech: {speech.shape}")
    print(f"  speech_lengths: {speech_lengths.shape}")
    print(f"Original output shapes:")
    print(f"  encoder_out: {encoder_out_original.shape}")
    print(f"  encoder_out_lens: {encoder_out_lens_original.shape}")
    
    # 将输入移到正确的设备
    device = next(audio_encoder.parameters()).device
    speech = speech.to(device)
    speech_lengths = speech_lengths.to(device)
    
    # 使用 audio_encoder 进行推理
    # 注意：encode 方法中会将 speech 从 [b, d, T] permute 到 [b, T, d]
    print("\nRunning inference with audio_encoder...")
    with torch.no_grad():
        # 根据 encode 方法的实现，需要 permute
        encoder_out_new, encoder_out_lens_new = audio_encoder(
            speech.permute(0, 2, 1), speech_lengths
        )
    
    print(f"New output shapes:")
    print(f"  encoder_out: {encoder_out_new.shape}")
    print(f"  encoder_out_lens: {encoder_out_lens_new.shape}")
    
    # 将原始输出也移到相同设备以便比较，并确保数据类型一致
    encoder_out_original = encoder_out_original.to(device).to(encoder_out_new.dtype)
    encoder_out_lens_original = encoder_out_lens_original.to(device)
    
    # 简单比较结果差异
    print("\n" + "="*60)
    print("Comparing results...")
    print("="*60)
    
    # 比较 encoder_out
    if encoder_out_original.shape == encoder_out_new.shape:
        mae = torch.mean(torch.abs(encoder_out_original - encoder_out_new)).item()
        is_match = torch.allclose(encoder_out_original, encoder_out_new, rtol=1e-5, atol=1e-8)
        print(f"\nEncoder Output:")
        print(f"  MAE: {mae:.6e}")
        print(f"  Match: {'✓ Yes' if is_match else '✗ No'}")
    else:
        print(f"\nEncoder Output: Shape mismatch!")
        print(f"  Original: {encoder_out_original.shape}, New: {encoder_out_new.shape}")
    
    # 比较 encoder_out_lens
    if torch.equal(encoder_out_lens_original, encoder_out_lens_new):
        print(f"\nEncoder Output Lengths: ✓ Match")
    else:
        print(f"\nEncoder Output Lengths: ✗ Not match")
        print(f"  Original: {encoder_out_lens_original}, New: {encoder_out_lens_new}")
    
    print("="*60)