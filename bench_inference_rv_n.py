"""Latency vs number of control points N — reviewer response, all five environments.

Same convention as the other bench_inference_* scripts: random weights (latency
depends on shapes and the graph, not weight values), warm-up, CUDA-synchronised
timing, batch 1 = one env step. Each environment's architecture is read from the
reference line of its reviewer batch (batches/rv*.txt, or q3cParticle16gpoff.txt
for particle), so the benchmark times exactly the networks those jobs train.
Only N (and top_k, clamped to N) varies.

Eval mode timed is the one each environment is scored with: argmax everywhere,
plus Langevin-50 for kitchen (its reference eval). The flat-state Langevin path
starts one chain per control point, so its cost grows with N.

    uv run --managed-python --extra libero --extra pushing python bench_inference_rv_n.py
"""
from __future__ import annotations

import argparse, csv, json, shlex, statistics, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
CFG = json.load(open(ROOT / "config_json/config.json"))
GRIDS = {"particle": [1, 2, 3, 5, 8, 10, 20], "pen": [1, 5, 20, 50, 100, 200], "kitchen": [1, 50, 100, 200],
         "pushing_pixels": [1, 2, 5, 8, 10, 20, 50, 100], "libero_goal_pixels": [1, 2, 5, 8, 10, 20, 50, 100]}
SRC = {"particle": "q3cParticle16gpoff.txt", "pen": "rvPen.txt", "kitchen": "rvKitchen.txt",
       "pushing_pixels": "rvPushingPixels.txt", "libero_goal_pixels": "rvLibero.txt"}


def ref_params(env):
    for line in open(ROOT / "batches" / SRC[env]):
        if line.startswith("uv run"):
            tok = shlex.split(line); return json.loads(tok[tok.index("--fixed-params") + 1])


def timeit(fn, dev, steps, warm):
    for _ in range(warm): fn()
    if dev.type == "cuda": torch.cuda.synchronize()
    t = []
    for _ in range(steps):
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); fn()
        if dev.type == "cuda": torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(t), statistics.median(t), statistics.stdev(t)


def flat_builders(env, p, dev):
    from utils.models import ControlPointGenerator, QEstimator
    e = CFG["environments"][env]
    fs = int(p.get("frame_stack", e.get("frame_stack", 1)))
    obs_dim = int(p.get("state_dim", e["state_dim"])) * fs
    act = int(p.get("action_dim", e["action_dim"])) * int(p.get("action_chunk", 1) or 1)
    cpw, cpd = int(p.get("cp_width", 256)), int(p.get("cp_depth", 2)); qw, qd = int(p.get("q_width", 256)), int(p.get("q_depth", 2))
    q = QEstimator(state_dim=obs_dim, action_dim=act, hidden_dims=[qw] * qd, network_kind=p.get("q_network_kind", "mlp"), width=qw, depth=qd,
                   use_spectral_norm=bool(p.get("q_use_spectral_norm", False)),
                   resnet_final_activation=bool(p.get("q_resnet_final_activation", True))).to(dev).eval()
    obs = torch.randn(1, obs_dim, device=dev)
    lo, hi = e.get("action_bounds", [-1, 1])

    def make(N):
        g = ControlPointGenerator(obs_dim, act, control_points=N, hidden_dims=[cpw] * cpd, action_bounds=(lo, hi),
                                  network_kind=p.get("cp_network_kind", "mlp"), width=cpw, depth=cpd,
                                  output_activation=p.get("cp_output_activation", "tanh")).to(dev).eval()
        @torch.no_grad()
        def argmax():
            c = g(obs); return c[0, q(obs.unsqueeze(1).expand(-1, N, -1), c).squeeze(-1).argmax(1)[0]]
        out = {"argmax": argmax}
        if env == "kitchen":
            from utils.sampling import sample_langevin
            amin = torch.full((act,), float(lo), device=dev); amax = torch.full((act,), float(hi), device=dev)
            for prm in q.parameters(): prm.requires_grad_(False)
            def langevin50():
                with torch.no_grad(): c = g(obs)
                ref = sample_langevin(lambda o, a: -q(obs.unsqueeze(1).expand(-1, a.shape[1], -1), a).squeeze(-1),
                                      obs, num_samples=N, action_min=amin, action_max=amax, num_iterations=50,
                                      initial_actions=c.clone(), device=dev)
                with torch.no_grad(): return ref[0, q(obs.unsqueeze(1).expand(-1, N, -1), ref).squeeze(-1).argmax(1)[0]]
            out["langevin50"] = langevin50
        return out, sum(x.numel() for x in g.parameters()) + sum(x.numel() for x in q.parameters())
    return make


def pixel_builders(env, p, dev):
    from utils.models import PixelControlPointGenerator, PixelQEstimator
    e = CFG["environments"][env]; m = e.get("model", {})
    lib = env == "libero_goal_pixels"
    if lib:
        import bench_inference_libero as BL
        in_ch, hw, act, cond, goal = BL.IN_CHANNELS, BL.CROP, BL.ACTION_DIM, BL.COND_DIM, BL.GOAL_DIM
        enc = dict(encoder_target_height=BL.ENC_HW, encoder_target_width=BL.ENC_HW, encoder_kind=p.get("encoder_kind", "resnet18"),
                   encoder_pretrained=False, encoder_num_kp=int(p.get("encoder_num_kp", 128)), encoder_norm_kind=p.get("encoder_norm_kind", "gn"),
                   encoder_per_camera=False, cond_fusion=p.get("cond_fusion", "film"), goal_dim=goal)
    else:
        sd = e["state_dim"]; in_ch = int(sd[0]) * int(p.get("frame_stack", e.get("frame_stack", 1))); hw = (int(sd[1]), int(sd[2]))
        act = int(e["action_dim"]) * int(p.get("action_chunk", 1) or 1); cond, goal = 0, 0
        enc = dict(encoder_target_height=int(e.get("encoder_target_height", 180)), encoder_target_width=int(e.get("encoder_target_width", 240)),
                   encoder_kind=p.get("encoder_kind", "conv_maxpool"), encoder_pretrained=False)
    H, W = (hw, hw) if isinstance(hw, int) else hw
    obs = torch.randint(0, 255, (1, in_ch, H, W), dtype=torch.uint8, device=dev)
    cond_t = torch.randn(1, cond, device=dev) if cond else None
    q = PixelQEstimator(action_dim=act, in_channels=in_ch, value_width=int(m.get("value_width", 1024)),
                        value_num_blocks=int(m.get("value_num_blocks", 1)), cond_dim=cond, **enc).to(dev).eval()
    if cond_t is not None: q._cond = cond_t
    lo, hi = e.get("action_bounds", [-1, 1])

    def make(N):
        cpw, cpd = int(p.get("cp_width", 256)), int(p.get("cp_depth", 2))
        g = PixelControlPointGenerator(output_dim=act, control_points=N, hidden_dims=[cpw] * cpd, action_bounds=(lo, hi),
                                       network_kind=p.get("cp_network_kind", "mlp"), width=cpw, depth=cpd, in_channels=in_ch,
                                       cond_dim=cond, output_activation=p.get("cp_output_activation", "tanh"), **enc).to(dev).eval()
        if cond_t is not None: g._cond = cond_t
        @torch.no_grad()
        def argmax():
            c = g(obs); f = q.encode(obs); return c[0, q.score(f, c).squeeze(-1).argmax(1)[0]]
        return {"argmax": argmax}, sum(x.numel() for x in g.parameters()) + sum(x.numel() for x in q.parameters())
    return make


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="+", default=list(GRIDS))
    ap.add_argument("--num-steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--csv", type=Path, default=ROOT / "results/reviewer/latency_vs_N.csv")
    a = ap.parse_args()
    dev = torch.device("cuda" if a.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); rows = []
    print(f"device={dev}  random weights; per-env architecture from its reviewer batch reference line")
    for env in a.envs:
        p = ref_params(env)
        make = (pixel_builders if env in ("pushing_pixels", "libero_goal_pixels") else flat_builders)(env, p, dev)
        for N in GRIDS[env]:
            fns, params = make(N)
            for mode, fn in fns.items():
                steps = a.num_steps if mode == "argmax" else max(20, a.num_steps // 10)
                mean, med, sd = timeit(fn, dev, steps, a.warmup)
                rows.append(dict(env=env, N=N, eval=mode, ms_mean=round(mean, 3), ms_median=round(med, 3), ms_std=round(sd, 3),
                                 params_m=round(params / 1e6, 3), device=dev.type))
                print(f"  {env:20} N={N:>4}  {mode:10} {mean:9.3f} ms  (median {med:8.3f}, sd {sd:6.3f})  params {params/1e6:6.2f}M", flush=True)
            del fns
            if dev.type == "cuda": torch.cuda.empty_cache()
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    with a.csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"-> {a.csv}")


if __name__ == "__main__":
    main()
