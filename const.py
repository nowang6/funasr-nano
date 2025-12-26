import torch

dtype_map = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32
}

dtypte = "bf16"

llm_dim = 1024
encoder_in_dim = 560
encoder_out_dim = 512

hotwords = ['张三疯', '警单']




encoder_conf = {
    'output_size': 512,
    'attention_heads': 4,
    'linear_units': 2048,
    'num_blocks': 50,
    'tp_blocks': 20,
    'dropout_rate': 0.1,
    'positional_dropout_rate': 0.1,
    'attention_dropout_rate': 0.1,
    'input_layer': 'pe',
    'pos_enc_class': 'SinusoidalPositionEncoder',
    'normalize_before': True,
    'kernel_size': 11,
    'sanm_shfit': 0,
    'selfattention_layer_type': 'sanm',
    'freeze': True,
    'freeze_layer_num': -1,
    'feat_permute': True
}

adaptor_conf = {
    'downsample_rate': 1,
    'ffn_dim': 2048,
    'llm_dim': 1024,
    'encoder_dim': 512,
    'n_layer': 2,
    'freeze': True
}

llm_conf = {
    'hub': 'hf',
    'freeze': True,
    'llm_dtype': 'bf16',
    'use_lora': False,
    'lora_conf': {
        'freeze_lora': True,
        'task_type': 'CAUSAL_LM',
        'r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'bias': 'none',
        'target_modules': ['q_proj', 'v_proj'],
        'init_param_path': ''
    }
}

MAX_NEW_TOKENS = 512

input_size = 560