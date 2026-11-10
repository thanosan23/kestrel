import json
from huggingface_hub import hf_hub_download
import struct

size = "large"

vocab_filename = hf_hub_download(repo_id=f"openai-community/gpt2-{size}", filename="vocab.json")
merges_filename = hf_hub_download(
    repo_id=f"openai-community/gpt2-{size}", filename="merges.txt"
)

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    n = 0
    cs = bs[:]
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

with open(vocab_filename, "r", encoding="utf-8") as f:
    encoder = json.load(f)

id_to_token = [None] * len(encoder)

for tok, i in encoder.items():
    id_to_token[i] = tok

def token_bytes(tok):
    return bytes(byte_decoder[c] for c in tok)

merges = []

with open(merges_filename, "r", encoding="utf-8") as f:
    next(f)  # skip the top line
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        a, b = line.split(" ")
        merges.append((encoder[a], encoder[b], encoder[a + b]))

byte_to_id = [encoder[byte_encoder[b]] for b in range(256)]

with open(f"gpt2_{size}_tokenizer.kestrel", "wb") as f:
    f.write(b"KTOK")
    f.write(struct.pack("<I", 1))
    f.write(struct.pack("<I", len(id_to_token)))
    for tok in id_to_token:
        bts = token_bytes(tok)
        f.write(struct.pack("<I", len(bts)))
        f.write(bts)
    f.write(struct.pack("<I", len(merges)))
    for a, b, m in merges:
        f.write(struct.pack("<3I", a, b, m))
    f.write(struct.pack("<256I", *byte_to_id))
