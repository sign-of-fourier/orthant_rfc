"""Run pooled-phase X1 splits on Modal CPU containers, one split per container, in parallel.

    modal run modal_x1.py --only 14,16,18

Each container runs `python x1.py run --only i` on a copy of the code and data and returns the split's
out/splits/NN.npz and NN.json, which are written locally; log lines are appended to out/x1.log.
1 physical core (2 vCPU) and 2 GiB per container, the same as the local box.
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REMOTE = "/root/analysis/X1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("r-base-core", "r-cran-bayesm")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy==2.5.1", "scipy==1.16.0", "xgboost==3.2.0", "pandas==2.2.3")
    .add_local_file(HERE / "core.py", f"{REMOTE}/core.py")
    .add_local_file(HERE / "x1.py", f"{REMOTE}/x1.py")
    .add_local_file(HERE / "out" / "data" / "camera_long.csv", f"{REMOTE}/out/data/camera_long.csv")
    .add_local_file(HERE.parent / "T1" / "hb.R", "/root/analysis/T1/hb.R")
)
app = modal.App("x1-splits", image=image)


@app.function(cpu=1.0, memory=2048, timeout=3600, retries=1)
def run_split(i: int) -> dict:
    import subprocess
    r = subprocess.run(["python", "x1.py", "run", "--only", str(i)], cwd=REMOTE, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"split {i}: {r.stderr[-3000:]}")
    out = Path(REMOTE) / "out"
    return dict(i=i, npz=(out / "splits" / f"{i:02d}.npz").read_bytes(),
                json=(out / "splits" / f"{i:02d}.json").read_text(), log=(out / "x1.log").read_text())


@app.local_entrypoint()
def main(only: str):
    idx = [int(x) for x in only.split(",")]
    todo = [i for i in idx if not (HERE / "out" / "splits" / f"{i:02d}.npz").exists()]
    print(f"{len(todo)} splits to run ({len(idx) - len(todo)} already present)")
    for res in run_split.map(todo, order_outputs=False, return_exceptions=True,
                              wrap_returned_exceptions=False):
        if isinstance(res, Exception):
            print(f"FAILED: {res}")
            continue
        d = HERE / "out" / "splits"
        (d / f"{res['i']:02d}.npz").write_bytes(res["npz"])
        (d / f"{res['i']:02d}.json").write_text(res["json"])
        with open(HERE / "out" / "x1.log", "a") as f:
            f.write("".join(f"{line} [modal]\n" for line in res["log"].splitlines()))
        print(res["log"].splitlines()[-1][:200], flush=True)
