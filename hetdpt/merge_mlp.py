from .models_pruned import MLP_no_act
import torch
def merge_mlp_model(model, current="", device="cpu"):
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        fqn = f"{current}.{name}" if current else name
        if isinstance(child, MLP_no_act):
            print(f"Merging MLP for {fqn}")
            input_dim = child.fc1.in_features
            output_dim = child.fc2.out_features
            merge_linear = torch.nn.Linear(input_dim, output_dim, bias=True)
            # load weights from the MLP
            fc1_weight = child.fc1.weight.data
            fc1_bias = child.fc1.bias.data if child.fc1.bias is not None else None
            fc2_weight = child.fc2.weight.data
            fc2_bias = child.fc2.bias.data if child.fc2.bias is not None else None
            fc_weight = torch.matmul(fc2_weight, fc1_weight)
            fc_bias = torch.matmul(fc1_bias, fc2_weight.t()) + fc2_bias 
            merge_linear.weight.data = fc_weight
            merge_linear.bias.data = fc_bias 
            setattr(model, name, merge_linear)
            del child
        else:
            merge_mlp_model(child, current=fqn, device=device)
    
def swap_activation(model, current="", device="cpu"):
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        fqn = f"{current}.{name}" if current else name
        if isinstance(child, torch.nn.GELU):
            print(f"Swapping activation for {fqn}")
            new_act = Learnable_Gelu_hard(device=device)
            setattr(model, name, new_act)
            del child
        else:
            swap_activation(child, current=fqn, device=device)