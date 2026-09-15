"""Consistency Policy inference latency per env step, for every cpv2 configuration.

Same convention as the other bench_inference_* scripts: random weights (latency depends
on shapes and the graph, not weight values), warm-up, CUDA-synchronised timing, batch
1 = one env step. Architectures come from results/reviewer/cp_architectures.json (the
cp_meta.json of the trained seed-0 checkpoints), so each model is exactly the one the
cpFlat / cpLibero / cpPushingPixels batches trained.

The timed call is the evaluation path, ConsistencyPolicyGenerator.forward: pixel
encoding (when the env has images) + the sampler + mapping the action back to its box.
Samplers:
  student : one jump sigma_max -> sigma_min           (1 network evaluation)
  chain3  : student + chaining D:27,54                (3 network evaluations)
  teacher : EMA teacher, Heun over the 80-bin grid    (network evaluations counted)
Not timed: environment step/rendering, host->device copy of the observation, CPU-side
observation normalisation.

    uv run --managed-python --extra libero --extra pushing python bench_inference_cp.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from utils.consistency import ConsistencyPolicyGenerator, CPModel, KarrasSchedule

ROOT = Path(__file__).resolve().parent
ARCH = ROOT / "results/reviewer/cp_architectures.json"
# Image size the policy receives (before its encoder resizes): LIBERO's random 116 crop,
# pushing pixels' native 240x320 frames.
IMAGE_HW = {"libero_goal_pixels": (116, 116), "pushing_pixels": (240, 320)}


def timeit(fn, dev, steps, warm):
    for _ in range(warm):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = []
    for _ in range(steps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(t), statistics.median(t), statistics.stdev(t)


def build(meta: dict, student: bool, dev) -> CPModel:
    common = dict(cond_dim=int(meta.get("cond_dim") or 0), time_emb_dim=int(meta["time_emb_dim"]),
                  network_kind=meta["network_kind"], width=int(meta["width"]), depth=int(meta["depth"]),
                  two_times=student, dropout=float(meta["dropout"]) if student else 0.0)
    if meta["pixel"]:
        ek = dict(meta.get("encoder_kwargs") or {})
        ek["encoder_pretrained"] = False  # random weights; never download
        model = CPModel(int(meta["action_dim"]), in_channels=int(meta["in_channels"]), encoder_kwargs=ek, **common)
    else:
        model = CPModel(int(meta["action_dim"]), state_dim=int(meta["state_dim"]), **common)
    return model.to(dev).eval()


class CountingHead(torch.nn.Module):
    """Wraps a CPHead to count network evaluations per sampled action."""

    def __init__(self, head):
        super().__init__()
        self.head, self.calls = head, 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.head(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.head, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="+", default=None)
    ap.add_argument("--num-steps", type=int, default=300, help="timed steps for student / chain3")
    ap.add_argument("--teacher-steps", type=int, default=30, help="timed steps for the teacher (158+ NFE each)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--csv", type=Path, default=ROOT / "results/reviewer/latency_cp.csv")
    a = ap.parse_args()
    dev = torch.device("cuda" if a.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    archs = {k: v for k, v in json.load(open(ARCH)).items() if not k.startswith("_")}
    envs = a.envs or list(archs)
    rows = []
    print(f"device={dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})  random weights, batch 1")
    for env in envs:
        meta = archs[env]
        sched = KarrasSchedule(sigma_min=meta["sigma_min"], sigma_max=meta["sigma_max"], rho=meta["rho"],
                               bins=meta["bins"], sigma_data=meta["sigma_data"])
        A, bounds = int(meta["action_dim"]), tuple(meta["action_bounds"])
        if meta["pixel"]:
            h, w = IMAGE_HW[env]
            obs = torch.randint(0, 256, (1, int(meta["in_channels"]), h, w), dtype=torch.uint8, device=dev)
        else:
            obs = torch.randn(1, int(meta["state_dim"]), device=dev)
        cond = torch.randn(1, int(meta["cond_dim"]), device=dev) if meta.get("cond_dim") else None
        student, teacher = build(meta, True, dev), build(meta, False, dev)
        samplers = [("student", student, "student", None, a.num_steps),
                    ("chain3", student, "student", meta.get("chaining_default") or "D:27,54", a.num_steps),
                    ("teacher", teacher, "teacher", None, a.teacher_steps)]
        for name, model, mode, chaining, steps in samplers:
            gen = ConsistencyPolicyGenerator(model, sched, A, bounds, mode=mode, chaining=chaining).to(dev).eval()
            gen._cond = cond
            real_head = model.head
            counter = CountingHead(real_head)
            model.head = counter
            with torch.no_grad():
                gen(obs.float())
            nfe = counter.calls
            model.head = real_head
            fn = lambda gen=gen: gen(obs.float())
            with torch.no_grad():
                mean, med, sd = timeit(fn, dev, steps, a.warmup if name != "teacher" else max(3, a.warmup // 10))
            params = sum(p.numel() for p in model.parameters())
            rows.append(dict(env=env, sampler=name, nfe=nfe, ms_mean=round(mean, 3), ms_median=round(med, 3),
                             ms_std=round(sd, 3), params_m=round(params / 1e6, 3), device=dev.type))
            print(f"  {env:20} {name:8} NFE={nfe:<4} {mean:9.3f} ms  (median {med:8.3f}, sd {sd:6.3f})  params {params / 1e6:6.2f}M",
                  flush=True)
        del student, teacher
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with a.csv.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"-> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
