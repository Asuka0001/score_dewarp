"""Summarize the DeepScores pretrained-vs-scratch ablation (outputs/deepscores_finetune).

Reads every run's training report, its evaluate_geometry results (eval_val / eval_test)
and the classical baselines, and writes a Markdown summary with paired per-page tests.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def read(path):
    return json.loads(Path(path).read_text())


def fmt(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


def per_page(eval_dir, metric="ad_pct"):
    return {r["sample"]: r.get(metric) for r in read(eval_dir / "per_sample.json")}


def paired(a_dir, b_dir, metric="ad_pct"):
    a, b = per_page(a_dir, metric), per_page(b_dir, metric)
    keys = sorted(k for k in a if k in b and a[k] is not None and b[k] is not None)
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    p = wilcoxon(x, y).pvalue if len(keys) > 5 and np.any(x != y) else None
    return {"n": len(keys), "median_diff": float(np.median(x - y)), "a_wins": int((x < y).sum()), "p": p}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs/deepscores_finetune"))
    ap.add_argument("--baselines", type=Path, default=Path("outputs/baselines"))
    ap.add_argument("--pairs", nargs="*", default=[], metavar="FT:SCRATCH",
                    help="run-name pairs for paired per-page tests")
    ap.add_argument("--out", type=Path, default=Path("reports/deepscores_finetune.md"))
    args = ap.parse_args()

    runs = sorted(p.parent for p in args.root.glob("*/best.report.json"))
    lines = ["## 训练记录（验证集选模指标：UV EPE %，越低越好）", "",
             "| 运行 | 初始化 | 训练样本 | LR | seed | 轮数/步数 | 初始 EPE | 最佳 EPE | 最佳轮 | 页面 IoU | 耗时 min |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        rep, cfg = read(run / "best.report.json"), read(run / "best.config.json")
        lines.append(
            f"| {run.name} | {'Doc3D' if cfg['init'] else '随机'} | {cfg['train_samples']} | {cfg['lr']:g} | {cfg['seed']} | "
            f"{rep['epochs_run']}/{rep['steps']} | {fmt(rep['initial_validation']['coord_epe_pct'])} | "
            f"**{fmt(rep['best_validation']['coord_epe_pct'])}** | {rep['best_epoch']} | "
            f"{fmt(rep['best_validation']['page_iou'], 4)} | {rep['elapsed_s'] / 60:.1f} |")

    for split in ("val", "test"):
        base_dir = args.baselines / f"deepscores_{split}"
        evaluated = [r for r in runs if (r / f"eval_{split}" / "summary.json").exists()]
        if not evaluated:
            continue
        lines += ["", f"## DeepScores {split}（{read(base_dir / 'config.json')['samples'].__len__()} 页）", "",
                  "| 方法 | AD % 中位数 | 局部畸变 % 中位数 | 平均覆盖率 | 全 GT 点 EPE % 均值 | 方向错误率 均值 | 失败 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        base_cfg = read(base_dir / "config.json")
        rows = []
        if (base_dir / "summary.json").exists():
            rows += [(k, v) for k, v in read(base_dir / "summary.json").items()]
        for run in evaluated:
            cfg = read(run / f"eval_{split}" / "config.json")
            assert cfg["samples"] == base_cfg["samples"], f"{run.name}: sample list differs from baseline"
            rows.append((run.name, read(run / f"eval_{split}" / "summary.json")["network"]))
        for name, s in rows:
            ori = s.get("orientation_error_rate_mean")
            lines.append(f"| {name} | {fmt(s['ad_pct_median'])} | {fmt(s['local_distortion_pct_median'])} | "
                         f"{s['coverage_mean'] * 100:.2f}% | {fmt(s.get('coord_epe_pct_all_gt_mean'))} | "
                         f"{'—' if ori is None else f'{ori * 100:.3f}%'} | {s['failed']} |")
        tests = []
        for pair in args.pairs:
            a, b = pair.split(":")
            if (args.root / a / f"eval_{split}").exists() and (args.root / b / f"eval_{split}").exists():
                r = paired(args.root / a / f"eval_{split}", args.root / b / f"eval_{split}")
                p = "—" if r["p"] is None else f"{r['p']:.2g}"
                tests.append(f"| {a} vs {b} | {r['n']} | {r['median_diff']:+.2f} | {r['a_wins']}/{r['n']} | {p} |")
        if tests:
            lines += ["", "逐页配对比较（AD %，差值 = 前者 − 后者，负数表示前者更好；Wilcoxon 符号秩检验）：", "",
                      "| 对比 | 页数 | 差值中位数 | 前者更好页数 | p |", "|---|---:|---:|---:|---:|", *tests]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
