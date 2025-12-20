import torch
from model import FunASRNano

model = FunASRNano(input_size=560)


ckpt = torch.load("models/Fun-ASR-Nano-2512/model.pt", map_location="cpu")
state_dict = ckpt["state_dict"]
    
    

# 去掉 DDP 的 module. 前缀（安全，不影响单卡）
state_dict = {
    k.replace("module.", ""): v
    for k, v in state_dict.items()
}

model.load_state_dict(state_dict)

# 3. 放到单卡
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. 推理模式
model.eval()

# 5. 推理
with torch.no_grad():
    output = model(input.to(device))