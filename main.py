from models.FunASRNano import FunASRNano

model_path = "weights/Fun-ASR-Nano-2512"

if __name__ == "__main__":

    model, kwargs = FunASRNano.from_pretrained(model_path=model_path, device="cuda:0", disalbe_update=True)
    model.eval()
    
    wav_path = f"data/创建警单.wav"
    res = model.inference(wav_path=wav_path, **kwargs)
    text = res[0][0]["text"]
    print(text)
