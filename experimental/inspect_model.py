"""load OLMoE and look at its structure. No hooks, no offloading yet."""
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import inspect

MODEL_ID = "allenai/OLMoE-1B-7B-0924"

print(f"torch {torch.__version__}, transformers {transformers.__version__}")

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()

cfg = model.config
print(f"layers={cfg.num_hidden_layers} experts={cfg.num_experts} "
      f"top_k={cfg.num_experts_per_tok} hidden={cfg.hidden_size} "
      f"expert_intermediate={cfg.intermediate_size}")

# Decode Layer (1/15)
print(model.model.layers[0])

# Identify where experts live
expert_params = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
total_params = sum(p.numel() for p in model.parameters())
print(f"total {total_params/1e9:.2f}B | experts {expert_params/1e9:.2f}B "
      f"({100*expert_params/total_params:.1f}%) | "
      f"non-expert {(total_params-expert_params)/1e9:.2f}B")
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/2**30:.1f} GiB")

mlp = model.model.layers[0].mlp
for name, p in mlp.named_parameters():
    print(f"{name:25s} {tuple(p.shape)}")
for cls in (type(mlp), type(mlp.gate), type(mlp.experts)):
    print(f"\n===== {cls.__name__}.forward =====")
    print(inspect.getsource(cls.forward))