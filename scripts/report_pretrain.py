"""Assemble the completed baseline / geometry pretraining experiment report."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("outputs/doc3d_pretrain"))
    ap.add_argument("--baselines", type=Path, default=Path("outputs/baselines"))
    ap.add_argument("--out", type=Path, default=Path("reports/geometry_pretrain.md"))
    args = ap.parse_args()
    def read(path):
        return json.loads(path.read_text())
    training = read(args.run / "best.report.json")
    config = read(args.run / "best.config.json")
    history = read(args.run / "best.history.json")
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    best = min(history[1:], key=lambda r: r["val_coord_epe_pct"])
    assert checkpoint["epoch"] == training["best_epoch"] == best["epoch"]
    assert abs(checkpoint["metrics"]["coord_epe_pct"] - best["val_coord_epe_pct"]) < 1e-6
    assert all(torch.isfinite(v).all() for v in checkpoint["model"].values())
    with (args.run / "best.pt").open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    summaries = {}
    for dataset in ("doc3d", "deepscores"):
        baseline_path = args.baselines / (dataset + "_val")
        model_path = args.run / ("eval_" + dataset + "_val")
        baseline_config, model_config = read(baseline_path / "config.json"), read(model_path / "config.json")
        assert baseline_config["data"] == model_config["data"]
        assert baseline_config["samples"] == model_config["samples"]
        assert model_config["checkpoint_sha256"] == sha
        summaries[dataset] = {**read(baseline_path / "summary.json"), **read(model_path / "summary.json")}
    initial = training["initial_validation"]["coord_epe_pct"]
    final = training["best_validation"]["coord_epe_pct"]
    lines = ["# 几何基线与 Doc3D 预训练结果", "",
             f"使用 Doc3D 第 8 分片的 {config['train_samples']} 张训练图和 {config['val_samples']} 张验证图，"
             f"在 {config['device_used']} 上运行 {training['epochs_run']} 轮、{training['steps']} 步，耗时 {training['elapsed_s']/60:.1f} 分钟。",
             f"输入为 {config['size'][0]}×{config['size'][1]}，batch={config['bs']}，初始学习率 {config['lr']}，seed={config['seed']}。",
             "仅训练几何坐标与页面掩码，冻结谱种、版面和结构头；本轮未使用测试集，也未做五线谱微调。", "",
             f"最佳权重来自第 {training['best_epoch']} 轮。验证 UV 坐标 EPE 从 {initial:.4f}% 降至 {final:.4f}%"
             f"（相对下降 {(1-final/initial)*100:.1f}%），页面 IoU 为 {training['best_validation']['page_iou']:.4f}。",
             "该 EPE 在所有 GT 页内像素上计算，不根据预测掩码筛点；两个坐标各归一化到 [0,1]。", "",
             "## 结果解读", "",
             f"Doc3D 上，模型的 AD 中位数为 {summaries['doc3d']['network']['ad_pct_median']:.2f}%，"
             f"原图为 {summaries['doc3d']['input']['ad_pct_median']:.2f}%，表明已学到同域文档几何校正能力。",
             f"未经乐谱微调时，五线谱验证集的模型 AD 为 {summaries['deepscores']['network']['ad_pct_median']:.2f}%，"
             f"高于原图的 {summaries['deepscores']['input']['ad_pct_median']:.2f}% 和 A+B 基线的 {summaries['deepscores']['A+B']['ad_pct_median']:.2f}%。"
             "预览可见谱线扭曲和局部折叠，当前权重应作为后续乐谱微调的初始化，实际乐谱校正先沿用传统基线。",
             "预训练对乐谱微调是否有收益，还需通过从零训练与预训练后微调的对照实验检验。", ""]
    labels = {"input": "原图", "A": "A 粗校正", "A+B": "A+B", "A+B+C": "A+B+C", "network": "Doc3D 预训练模型"}
    for dataset, methods in summaries.items():
        lines += [f"## {dataset} 验证集", "", "| 方法 | 页数 | AD % 中位数 | 局部畸变 % 中位数 | 平均覆盖率 | 失败 / AD未定义 |",
                  "|---|---:|---:|---:|---:|---:|"]
        def fmt(v):
            return "未定义" if v is None else f"{v:.4f}"
        for method, s in methods.items():
            lines.append(f"| {labels[method]} | {s['samples']} | {fmt(s['ad_pct_median'])} | "
                         f"{fmt(s['local_distortion_pct_median'])} | {s['coverage_mean']*100:.2f}% | {s['failed']} / {s['undefined_ad']} |")
        net = methods["network"]
        lines += ["", f"网络在所有 GT 页面采样点上的平均坐标 EPE（页宽单位）为 {net['coord_epe_pct_all_gt_mean']:.4f}%，"
                  f"采样单元方向错误率为 {net['orientation_error_rate_mean']*100:.3f}%。", ""]
    lines += ["## 指标与使用范围", "",
              "AD 在输入照片固定网格上比较映射，先以保方向相似变换将预测对齐到真值，再计算 RMS / 页宽。"
              "局部畸变比较同一对齐后相邻网格边向量的误差。它们与训练日志的未对齐 UV EPE 不同。",
              "AD/局部误差仅在有效预测点上统计，必须结合覆盖率和失败数阅读；不能把不同覆盖率下的低 AD 直接当作整体更好。",
              "训练验证 EPE 使用 256×256 训练目标；下面的稀疏网格评测从原始 GT 场采样，数值会有小幅差别。",
              "方向错误率用于发现采样网格中的折叠或塌缩，不保证整个稠密映射可逆。定性预览中的局部伪影仍需结合此指标判断。",
              "本轮全部为合成数据验证，不代表真实手机照片或下游 OMR 的识别精度。传统方法中无结构观测的崩溃已修复并重跑，原始记录保留。",
              "计时包含部分评测开销且执行设备/并发不同，不据此比较推理速度。", "",
              "## 产物", "",
              f"- 最佳权重：`{args.run / 'best.pt'}`；SHA-256：`{sha}`。",
              f"- 配置、训练曲线与逐轮日志：`{args.run}/best.config.json`、`best.curves.png`、`best.history.json`。",
              f"- 两个验证集的逐页结果：`{args.run}/eval_*_val/per_sample.json`。",
              f"- 模型校正预览：`{args.run}/preview_doc3d_val.jpg`、`preview_deepscores_val.jpg`。",
              f"- 传统基线：`{args.baselines}/doc3d_val/`、`{args.baselines}/deepscores_val/`。", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    args.out.with_suffix(".json").write_text(json.dumps({"training": training, "model_sha256": sha, "summaries": summaries}, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
