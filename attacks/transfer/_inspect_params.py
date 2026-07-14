import torch, os, collections
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from onnx2torch import convert
m = convert("../../checkpoints/model.onnx")
names = [n for n, _ in m.named_parameters()]
print("total param tensors:", len(names))
# coarse structure
pref = collections.Counter(n.split('.')[0].split('/')[0][:28] for n in names)
print("top-level prefixes:")
for k, v in pref.items():
    print(f"  {k}: {v}")
print("--- first 10 ---")
for n in names[:10]:
    print("  ", n)
print("--- last 10 ---")
for n in names[-10:]:
    print("  ", n)
# look for resnet-style layer markers
for key in ["layer1", "layer2", "layer3", "layer4", "stage1", "stage2", "stage3", "stage4", "conv1", "prelu", "fc", "bn", "Gemm", "features"]:
    c = sum(1 for n in names if key.lower() in n.lower())
    if c:
        print(f"contains '{key}': {c} tensors")
