import argparse
import os
import runpy
import sys
from pathlib import Path

COMMANDS = {
    "data": {
        "make-mask": ("insf.data.mask", "build the 601-band coded mask from the camera QE sheet and channel layout"),
        "convert": ("insf.data.convert", "convert raw camera frames (.qs/.raw) to .mat"),
        "make-holdout": ("insf.data.holdout", "regenerate the fixed 10 % test pixels of move2"),
        "prepare-colorchecker": ("insf.data.colorchecker", "copy the selected colour-chart frames and white-tile masks"),
    },
    "reconstruct": {
        "single-frame": ("insf.recon.single_frame", "per-frame INR on a sequence (E1), warm start (E2), space-time field (E3)"),
        "single-image": ("insf.recon.single_image", "INR on one coded image"),
        "shared-field": ("insf.recon.shared_field", "one field shared by the training frames of a static scene"),
        "mean-input": ("insf.recon.mean_input", "INR fitted to the temporal mean of the training frames"),
        "flow": ("insf.recon.flow", "DIS optical flow between single-frame reconstructions"),
        "mc-window": ("insf.recon.mc_window", "motion-compensated multiframe INR (or single-frame INR with held-out pixels)"),
        "static-capture": ("insf.recon.static_capture", "all methods on one static capture"),
        "colorchecker": ("insf.recon.colorchecker", "single-frame INR on a colour-chart capture"),
    },
    "baseline": {
        "twist": ("insf.baselines.twist", "TwIST with a TV prior"),
        "admm": ("insf.baselines.admm", "ADMM with a TV prior"),
        "mst": ("insf.baselines.mst", "train or apply MST / MST++ (--arch mst|mstpp)"),
        "mst-generalization": ("insf.baselines.mst_generalization", "apply an MST checkpoint to the simulated datasets"),
        "collect": ("insf.baselines.collect", "summarise the baseline runs"),
    },
    "simulate": {
        "generalization-prep": ("insf.simulation.generalization_prep", "load KAIST / CAVE / Harvard scenes and build the job queue"),
        "generalization": ("insf.simulation.generalization", "simulate coded frames and reconstruct them (one worker per GPU)"),
        "static-scene": ("insf.simulation.static_scene", "simulate a coded frame of a CAVE scene"),
        "motion": ("insf.simulation.motion", "controlled motion experiment with prescribed warps"),
    },
    "evaluate": {
        "mc-window": ("insf.evaluation.mc_window", "metrics, images and videos of the multiframe and single-frame runs"),
        "generalization": ("insf.evaluation.generalization_metrics", "metrics of the simulated datasets"),
        "generalization-render": ("insf.evaluation.generalization_render", "colour images of the simulated datasets"),
        "static-scene": ("insf.evaluation.static_scene", "metrics of the simulated static scene"),
        "colorchecker": ("insf.evaluation.colorchecker", "reflectance of the 24 patches against reference data"),
    },
}


def config_to_argv(path):
    import yaml
    cfg = yaml.safe_load(open(path)) or {}
    for k, v in (cfg.pop("env", None) or {}).items():
        os.environ.setdefault(k, str(v))
    argv = [str(x) for x in cfg.pop("positional", None) or []]
    cfg.pop("command", None)
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-") if not k.startswith("-") else k
        if v is None or v is False:
            continue
        if v is True:
            argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv += [flag] + [str(x) for x in v]
        else:
            argv += [flag, str(v)]
    return argv


def usage():
    lines = ["usage: python main.py <group> <command> [--config file.yaml] [arguments]", ""]
    for g, cmds in COMMANDS.items():
        lines.append(g)
        lines += [f"  {c:22s} {d}" for c, (_, d) in cmds.items()]
        lines.append("")
    lines.append("Arguments of a command: python main.py <group> <command> --help")
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] not in COMMANDS or argv[1] not in COMMANDS[argv[0]]:
        print(usage())
        return 0 if argv[:1] in (["-h"], ["--help"], []) else 2
    module = COMMANDS[argv[0]][argv[1]][0]
    rest = argv[2:]
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", type=Path)
    known, rest = ap.parse_known_args(rest)
    if known.config:
        rest = config_to_argv(known.config) + rest
    sys.argv = [module] + rest
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0
