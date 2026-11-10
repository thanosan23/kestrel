import json
from huggingface_hub import hf_hub_download
import numpy as np
import struct
import tqdm

size = "large"

filepath = hf_hub_download(
    repo_id=f"openai-community/gpt2-{size}", filename="model.safetensors"
)

weights = {}
# Load safetensors
with open(filepath, "rb") as f:
    contents = f.read()
    header_size = int.from_bytes(contents[:8], byteorder="little")
    header = json.loads(contents[8 : 8 + header_size].decode())

    header.pop("__metadata__", None)

    for tensor_name, info in tqdm.tqdm(header.items(), desc="Loading weights"):
        dtype = info["dtype"]
        shape = info["shape"]

        start_offset, end_offset = info["data_offsets"]

        data = contents[8 + header_size + start_offset : 8 + header_size + end_offset]

        if dtype == "F32":
            dtype = np.float32
        elif dtype == "F64":
            dtype = np.float64
        elif dtype == "I64":
            dtype = np.int64
        else:
            print(f"Unsupported datatype {dtype}")

        data = np.frombuffer(data, dtype=dtype).reshape(shape)

        weights[tensor_name] = data

with open(f"gpt2_{size}_weights.kestrel", "wb") as f:
    f.write(struct.pack("I", len(weights)))
    for tensor_name, data in weights.items():
        tensor_name_bytes = tensor_name.encode()
        f.write(struct.pack("<I", len(tensor_name_bytes)))
        f.write(tensor_name_bytes)
        f.write(struct.pack("<I", data.ndim))
        f.write(struct.pack(f"<{data.ndim}I", *data.shape))
        f.write(data.astype(np.float32).tobytes())
