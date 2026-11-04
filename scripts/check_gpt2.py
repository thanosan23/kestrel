import subprocess
import sys
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

KESTREL = "./bin/main"
DUMP = "/tmp/kestrel_dump.bin"
N_GEN = 30
LOGIT_TOLERANCE = 1e-2

PROMPTS = [
    "Hello world",
    "The history of the Roman Empire",
    "Once upon a time, in a land far away,",
    "I'll go  now! It's 3:45pm and we've got 1234 apples.",
    "Café crème, naïve résumé, 日本語"
]

def run_kestrel(args):
    result = subprocess.run([KESTREL, *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"Kestrel failed:\n{result.stderr}")
    return result.stdout

def read_dump(path):
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw, dtype=np.int32, count=1)[0])
    ids = np.frombuffer(raw, dtype=np.int32, count=n, offset=4).tolist()
    logits = np.frombuffer(raw, dtype=np.float32, offset=4 + 4 * n)
    return ids, logits

def first_different(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32).eval()

all_passed = True
for prompt in PROMPTS:
    print(f"\nPrompt: {prompt}")
    ref_ids = tok.encode(prompt)
    run_kestrel(["-p", prompt, "-n", "0", "--dump", DUMP])
    k_ids, k_logits = read_dump(DUMP)

    ids_ok = k_ids == ref_ids
    print(f" token ids: {"PASS" if ids_ok else "FAIL"}")
    if not ids_ok:
        print(f"    kestrel: {k_ids}")
        print(f"    reference: {ref_ids}")
        all_passed = False
        continue

    with torch.no_grad():
        ref_logits = model(torch.tensor([ref_ids])).logits[0, -1].numpy()

    max_diff = float(np.abs(k_logits - ref_logits).max())
    k_top5 = np.argsort(-k_logits)[:5].tolist()
    ref_top5 = np.argsort(-ref_logits)[:5].tolist()
    logits_ok = max_diff < LOGIT_TOLERANCE and k_top5 == ref_top5

    print(f" logits: {'PASS' if logits_ok else 'FAIL'}  "
          f"(max diff {max_diff:.2e}, top-5 {'match' if k_top5 == ref_top5 else 'differ'})")

    if k_top5 != ref_top5:
        print(f"    kestrel top 5:   {[tok.decode([i]) for i in k_top5]}")
        print(f"    reference top 5: {[tok.decode([i]) for i in ref_top5]}")

    all_passed = all_passed and logits_ok


print("\nALL_PASSED" if all_passed else "\nSOME CHECKS FAILED")
sys.exit(0 if all_passed else 1)
