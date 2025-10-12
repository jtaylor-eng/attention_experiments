# Drop in adaptive temperature softmax to pretrained Gemma2B

import torch
import torch.nn.functional as F
from transformers.models.gemma.modeling_gemma import GemmaAttention
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


class AdaptiveTemperatureAttn(GemmaAttention):
    """Drop adaptive_temperature_softmax in for F.softmax so model tunes using it in attention."""
    def forward(self, *args, **kwargs):
        orig_softmax = F.softmax

        F.softmax = adaptive_temperature_softmax #compute output using our attn
        out = super().forward(*args, **kwargs)

        F.softmax = orig_softmax #ensure softmax is set back to traditional

        return out

 
def adaptive_temperature_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Implements the adaptive temperature softmax from the paper translated to pytorch."""
    poly_fit = torch.tensor([-0.037, 0.481, -2.3, 4.917, -1.791], device=logits.device)
    
    #calculate initial probs & entropy
    with torch.no_grad():
        original_probs = F.softmax(logits, dim=-1)
        entropy = torch.sum(-original_probs * torch.log(original_probs + 1e-9), dim=-1, keepdim=True)

    #Gemini: TODO: check
    # Calculate beta (1/theta) based on the polynomial fit [cite: 224, 226]
    # The paper's JAX code uses polyval, which evaluates from the highest power.
    # PyTorch's polyval needs the tensor of powers, so we build it.
    pows = torch.arange(len(poly_fit) - 1, -1, -1, device=logits.device)
    entropy_pows = entropy ** pows
    beta = torch.sum(poly_fit * entropy_pows, dim=-1, keepdim=True)
    
    beta = torch.where(
        entropy > 0.5,
        torch.maximum(beta, torch.tensor(1.0, device=logits.device)),
        torch.tensor(1.0, device=logits.device)
    )
    
    return F.softmax(logits * beta, dim=-1, dtype=torch.float32).to(logits.dtype)


def swap_gemma_attention_layers(model):
    """Recursively traverses model and replaces each GemmaAttention w/ AdaptiveTempAttn"""
    for name, module in model.named_children():
        if isinstance(module, GemmaAttention): #is gemma attn, replace
            adaptive_layer = AdaptiveTemperatureAttn(
                config=module.config,
                layer_idx=module.layer_idx
            ).to(device=module.q_proj.weight.device, dtype=module.q_proj.weight.dtype)

            adaptive_layer.load_state_dict(module.state_dict())
            setattr(model, name, adaptive_layer)

        else: swap_gemma_attention_layers(module) #recurse
    return model

def test_gen(model, tokenizer):
    """Ensure functionality to generate from model w/ tokenizer."""
    input_text = 'Michael Jordan plays for the '
    input_ids = tokenizer(input_text, return_tensors='pt').to('cuda')
    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', device_map='auto', torch_dtype=torch.bfloat16)

    print('beginning swapping')
    model = swap_gemma_attention_layers(model)
    print('model swapped without errors')

    print('verifying model inference works')
    test_gen(model, tokenizer)
    print('model generated without errors')
