"""Stage 1a: load OLMoE and look at its structure. No hooks, no offloading yet."""
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
 
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
 
# One decoder layer is enough to see the pattern; the other 15 are identical.
print(model.model.layers[0])
 
# Where do the parameters live? Split expert weights from everything else.
expert_params = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
total_params = sum(p.numel() for p in model.parameters())
print(f"total {total_params/1e9:.2f}B | experts {expert_params/1e9:.2f}B "
      f"({100*expert_params/total_params:.1f}%) | "
      f"non-expert {(total_params-expert_params)/1e9:.2f}B")
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/2**30:.1f} GiB")
 
# Sanity check that the model actually produces sensible text.
inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(tok.decode(out[0]))