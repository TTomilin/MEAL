import argparse
import collections
import io
import json
import pickle
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.partner_adaptation.partner_agents.overcooked.concord_cnn import torch_state_dict_to_flax  # noqa: E402

RAW = "https://raw.githubusercontent.com/chenf-ai/Concord/main/CRL_human_AI_overcooked/human_models/model_random1"
NUM_MEMBERS = 12  # Concord App. B.2: 8 MEP + 4 HSP teammates per layout
LAYOUT = "coord_ring"
_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
           "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_}


def read_state_dict(path: Path, prefer_torch: bool = True):
    if prefer_torch:
        try:
            import torch
            return {k: v.detach().cpu().numpy() for k, v in torch.load(path, map_location="cpu").items()}
        except ImportError:
            pass
    z = zipfile.ZipFile(path)
    root = next(n for n in z.namelist() if n.endswith("data.pkl")).rsplit("/", 1)[0]

    def rebuild(storage, offset, size, stride, *_):
        dtype, key = storage
        buf = np.frombuffer(z.read(f"{root}/data/{key}"), dtype=dtype)
        return np.lib.stride_tricks.as_strided(buf[offset:], shape=tuple(size),
                                               strides=tuple(s * buf.itemsize for s in stride)).copy()

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "OrderedDict":
                return collections.OrderedDict
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name in _DTYPES:
                return _DTYPES[name]
            raise pickle.UnpicklingError(f"unexpected global {module}.{name} in {path}")

        def persistent_load(self, pid):  # ('storage', storage_type, key, location, numel)
            return pid[1], pid[2]

    return dict(Unpickler(io.BytesIO(z.read(f"{root}/data.pkl"))).load())


def fetch(src: Path, i: int) -> Path:
    path = src / f"actor_best_r{i}.pt"
    if not path.is_file():
        src.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{RAW}/actor_best_r{i}.pt", path)
    return path


def convert(out: Path, src: Path = None) -> Path:
    out = Path(out)
    src = Path(src) if src else out / "src"
    out.mkdir(parents=True, exist_ok=True)
    members = []
    for i in range(1, NUM_MEMBERS + 1):
        params = torch_state_dict_to_flax(read_state_dict(fetch(src, i)))
        fname = f"r{i}.pt"
        with open(out / fname, "wb") as f:
            pickle.dump({"actor_params": params}, f)
        members.append(dict(member_index=i - 1, seed_index=0, file=fname, concord_id=f"r{i}",
                            performance_bounds={}))
    gen = dict(status="complete", kind="concord", source="github.com/chenf-ai/Concord", run=f"{RAW}/actor_best_r*.pt",
               layout_name=LAYOUT, concord_layout="random1", actor_type="concord_cnn", partner_pop_size=1,
               num_members=NUM_MEMBERS, activation="relu", fc_hidden_dim=64, test_mode=False,
               obs_convention="overcooked_ai_lossless_old", trained_seat="both", members=members)
    (out / "generation.json").write_text(json.dumps(gen, indent=2) + "\n")
    return out / "generation.json"


def main():
    parser = argparse.ArgumentParser(
        description="Convert Concord's 12 MEP/HSP coord_ring teammates into MEAL partner-bank members.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--src", type=Path, default=None, help="directory with actor_best_r*.pt (downloaded if absent)")
    args = parser.parse_args()
    print(f"wrote {convert(args.out, args.src)}")


if __name__ == "__main__":
    main()
