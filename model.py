import torch

old_model = torch.load('epoch_5.pth')
print("None")

old_model['state_dict'] = {key: value for key, value in old_model['state_dict'].items() if 'offset' in key}

torch.save(old_model, "epoch_5_new.pth")
