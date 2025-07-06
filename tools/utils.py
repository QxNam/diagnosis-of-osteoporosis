from torchsummary import summary

def get_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"📊 Total parameters: {total_params:,}")
    # print(f"🧠 Trainable parameters: {trainable_params:,}")
    return total_params
    
def export_args(path:str, metadata:dict):
    with open(path, "w", encoding='utf-8') as f:
        for k, v in metadata.items():
            f.write(f'{k}: {v}\n')