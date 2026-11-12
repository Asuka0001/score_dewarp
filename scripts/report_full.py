"""Summarize the full-data model against the pilot model and classical baselines.

Every comparison is per page on an identical sample list (asserted from each
evaluation's config.json); paired differences use the Wilcoxon signed-rank test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

FT = Path("outputs/deepscores_finetune")
FULL, PILOT = FT / "full_ca_lr1e-3_s7", FT / "ft_ca_lr1e-3_s7"
BASE = Path("outputs/baselines")
# (title, baselines, pilot-model eval, full-model eval)
SETS = [
    ("pilot val（64 页，选模集的子集）", BASE / "deepscores_val", PILOT / "eval_val", FULL / "eval_val"),
    ("pilot test（64 页）", BASE / "deepscores_test", PILOT / "eval_test", FULL / "eval_test"),
    ("全量 test（137 页）", BASE / "deepscores_full_test", PILOT / "eval_full_test", FULL / "eval_full_test"),
]
LOW_COVERAGE = 0.8


def read(path):
    return json.loads(Path(path).read_text())


def fmt(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


def rows_by_method(eval_dir):
    out = {}
    for r in read(eval_dir / "per_sample.json"):
        out.setdefault(r["method"], {})[r["sample"]] = r
    return out


def paired(a, b, key):
    keys = sorted(k for k in a.keys() & b.keys() if a[k].get(key) is not None and b[k].get(key) is not None)
    x, y = np.array([a[k][key] for k in keys]), np.array([b[k][key] for k in keys])
    p = wilcoxon(x, y).pvalue if len(keys) > 5 and np.any(x != y) else None
    return len(keys), float(np.median(x - y)), int((x < y).sum()), p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("reports/deepscores_full.md"))
    args = ap.parse_args()

    rep, cfg = read(FULL / "best.report.json"), read(FULL / "best.config.json")
    hist = read(FULL / "best.history.json")
    lines = ["# 全量 DeepScores 训练：验证与测试", "",
             f"权重 `{FULL}/best.pt`：Doc3D 初始化，CoordConv + attention，384×512，LR {cfg['lr']:g}，seed {cfg['seed']}，"
             f"train {cfg['train_samples']} / val {cfg['val_samples']}。",
             f"按全量 val 的 UV EPE 选模：初始 {fmt(rep['initial_validation']['coord_epe_pct'])}% → "
             f"最佳 **{fmt(rep['best_validation']['coord_epe_pct'])}%**（第 {rep['best_epoch']} 轮，"
             f"共 {rep['epochs_run']} 轮 / {rep['steps']} 步，{rep['elapsed_s'] / 3600:.1f} h，停止原因 {rep['stop_reason']}）。",
             f"对比的 pilot 模型 `{PILOT}/best.pt` 仅用 128 个训练样本，其余设置相同。", ""]
    if rep["best_epoch"] == rep["epochs_run"] and rep["stop_reason"] == "early_stopping":
        last = [r["val_coord_epe_pct"] for r in hist[-6:]]
        lines += [f"注意：最佳轮就是最后一轮。`min_delta={cfg['min_delta']}` 是绝对阈值，EPE 降到约 0.55% 后每轮改进只有约 0.01，"
                  f"不再计为显著改进，因此在余弦学习率结束前触发早停（最后 6 轮 EPE："
                  f"{', '.join(f'{v:.3f}' for v in last)}）。模型可能还没有完全收敛。", ""]

    lines += ["测试集只评测一次，模型和 C 的参数都已经按验证集固定，没有根据 test 结果做任何调整。",
              "pilot val/test 与全量数据中对应页面的照片逐字节相同，各集合按曲谱分组划分，与全量 train 没有重叠。", ""]

    for title, base_dir, pilot_dir, full_dir in SETS:
        if not all((d / "summary.json").exists() for d in (base_dir, pilot_dir, full_dir)):
            lines += [f"## {title}", "", "（评测尚未完成）", ""]
            continue
        samples = read(base_dir / "config.json")["samples"]
        for d in (pilot_dir, full_dir):
            assert read(d / "config.json")["samples"] == samples, f"{d}: sample list differs from {base_dir}"
        base, pilot, full = rows_by_method(base_dir), rows_by_method(pilot_dir), rows_by_method(full_dir)
        table = [(m, base[m]) for m in base] + [("pilot 网络", pilot["network"])]
        table += [({"network": "全量网络", "network_bmap": "全量网络反演映射", "network+C": "全量网络+C"}[m], full[m])
                  for m in ("network", "network_bmap", "network+C") if m in full]
        lines += [f"## {title}", "",
                  f"| 方法 | AD % 中位数 | AD % 均值 | 局部畸变 % 中位数 | 平均覆盖率 | 覆盖率 <{LOW_COVERAGE:.0%} 页数 | 失败 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, rows in table:
            ok = [r for r in rows.values() if r["status"] == "ok"]
            ad = [r["ad_pct"] for r in ok if r.get("ad_pct") is not None]
            ld = [r["local_distortion_pct"] for r in ok if r.get("local_distortion_pct") is not None]
            cov = [r["coverage"] for r in rows.values()]
            bold = "**" if name == "全量网络" else ""
            lines.append(f"| {bold}{name}{bold} | {fmt(np.median(ad), 3)} | {fmt(np.mean(ad), 3)} | {fmt(np.median(ld), 2)} | "
                         f"{np.mean(cov) * 100:.2f}% | {sum(c < LOW_COVERAGE for c in cov)} | {len(rows) - len(ok)} |")
        lines += ["", "逐页配对比较（差值 = 前者 − 后者，负数表示前者更好；Wilcoxon 符号秩检验）：", "",
                  "| 对比 | 指标 | 页数 | 差值中位数 | 前者更好页数 | p |", "|---|---|---:|---:|---:|---:|"]
        comps = [("全量网络 vs A+B", full["network"], base["A+B"]),
                 ("全量网络 vs pilot 网络", full["network"], pilot["network"])]
        for label, a, b in comps:
            for key, name in (("ad_pct", "AD"), ("local_distortion_pct", "局部畸变")):
                n, d, wins, p = paired(a, b, key)
                lines.append(f"| {label} | {name} | {n} | {d:+.3f} | {wins}/{n} | {'—' if p is None else f'{p:.2g}'} |")
        if "network+C" in full:
            for key, name in (("common_ad_pct", "AD（共同有效点）"), ("common_local_distortion_pct", "局部畸变（共同有效点）")):
                n, d, wins, p = paired(full["network+C"], full["network_bmap"], key)
                lines.append(f"| 全量网络+C vs 全量网络反演映射 | {name} | {n} | {d:+.3f} | {wins}/{n} | {'—' if p is None else f'{p:.2g}'} |")
        lines.append("")

    lines += ["说明：A+B 的误差只在其有效点上计算，覆盖率明显更低，单看 AD / 局部畸变会偏向 A+B。",
              "C 的对比使用每页两种映射都有效的点；网络+C 尚未接入 B，所以结果不代表完整的网络+B+C 方案。",
              "所有形变都是合成的，尚未在真实手机照片上验证。"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
