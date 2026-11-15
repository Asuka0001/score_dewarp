"""Summarize the frozen synthetic benchmark and data/consistency ablation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def fmt(value, factor=1, digits=3):
    return "—" if value is None else f"{value * factor:.{digits}f}"


def paired(rows, left, right, kind=None, profile=None):
    """Paired *per-page differences*, each on the same two-method support."""
    group = {r["sample"]: r for r in rows if r["method"] == right and
             (kind is None or r["kind"] == kind) and (profile is None or r["profile"] == profile)}
    deltas = {key: [] for key in ("ad_pct", "local_distortion_pct", "staff_straight", "jp_baseline")}
    for a in rows:
        if a["method"] != left or a["sample"] not in group:
            continue
        b = group[a["sample"]]
        ma, mb = a.get("paired", {}).get(right, {}), b.get("paired", {}).get(left, {})
        for key in deltas:
            section = "geometry" if key in ("ad_pct", "local_distortion_pct") else "music"
            va, vb = ma.get(section, {}).get(key), mb.get(section, {}).get(key)
            if va is not None and vb is not None and np.isfinite([va, vb]).all():
                deltas[key].append(va - vb)
    return {key: {"n": len(values), "median_delta": median(values),
                  "improved": sum(v < 0 for v in values)} for key, values in deltas.items()}


def preview(experiment, suite, baseline, rows, destination):
    """One worst-baseline-AD test page per kind; no cherry-picked improvements."""
    import cv2
    chosen, strips = [], []
    for kind in ("staff", "jianpu", "mixed"):
        row = max((r for r in rows if r["kind"] == kind and r["method"] == baseline
                   and r.get("ad_pct") is not None), key=lambda r: r["ad_pct"])
        sample = row["sample"]
        result = experiment / "comparison_test" / suite.name / "test" / sample
        images = [("flat reference", suite / "test" / sample / "flat.png"),
                  ("input", suite / "test" / sample / "photo.jpg")]
        images += [(title, result / (method + "_rectified.png")) for title, method in
                   (("traditional", "traditional"), ("original net", baseline),
                    ("data only", "data_only"), ("+ consistency", "structure_consistency"))]
        tiles = []
        for title, path in images:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Missing preview image: {path}")
            tile = cv2.resize(image, (250, 354), interpolation=cv2.INTER_AREA)
            tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(tile, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 1)
            tiles.append(tile)
        strip = cv2.hconcat(tiles)
        strip = cv2.copyMakeBorder(strip, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(strip, f"{sample}: worst original-net AD within {kind} test pages",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
        strips.append(strip)
        chosen.append({"sample": sample, "kind": kind, "original_ad_pct": row["ad_pct"]})
    if not cv2.imwrite(str(destination), cv2.vconcat(strips), [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise ValueError(f"Cannot write {destination}")
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.experiment = args.experiment.resolve()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((args.experiment / "protocol.json").read_text())
    suite = root / protocol["suite"]
    manifest = json.loads((suite / "manifest.json").read_text())
    baseline = Path(protocol["baseline"]).parent.name
    methods = ["traditional", baseline, *protocol["arms"]]
    lines = ["# 固定合成基准与简谱/混排结构一致性训练（2026-11-15）", "",
             "本轮先建立分层合成基准，再完成两组受控微调；所有素材与形变均为合成。", "",
             "## 固定数据与实验协议", "",
             "- 新素材共 480 张：train/val/test = 300/90/90；每个集合内三种谱种均衡。",
             "- 五档条件：轻度形变（strength=0.5）、标准（1.0）、强形变（1.3）、强制投射阴影、固定模糊 sigma=1.8。后两档仍使用 strength=1.0；其他降质保持随机。",
             "- 每个 val/test 的“谱种 × 条件”格子有 6 张；每类共 30 张。每页独立随机种子、来源分组及文件 SHA-256，无重复平整页或跨集合来源重叠。",
             "- 生成时在 97×129 网格的每格两个三角形检查投影方向，最小有符号面积/中位面积需大于 0.02；失败则重采样形变，保留平整页内容。该采样检查不等于连续可逆性的数学证明。",
             "- 模拟照片为 768×1024；本轮统一输出宽度 1240。不同于此前宽度 1600 的结果，不直接跨报告比较检测指标。",
             "- 两组训练使用完全相同的 300 张新合成页和固定抽取的 100 张 DeepScores train 回放页。",
             "- 从既有全量权重初始化，输入 384×512，seed=7，batch=8，LR=1e-4，余弦下降，24 轮，无早停；各组按 90 张新 val 的平均逐页 UV EPE 保存最佳权重。",
             "- data_only：坐标、梯度、页面掩码损失；structure_consistency：额外加入权重 50 的结构实例一致性损失（水平结构的平面 Y 方差、竖直结构的平面 X 方差）。",
             "- 两组均冻结 seg/struct/kind 输出头，避免多任务监督混入对照；最终谱种仍来自手工检测器。该实验验证几何坐标的结构约束，尚未训练结构热图检测头。",
             "- 两组训练及选模完成后才评测保留 test；test 不用于调参。DeepScores 172 张 val 用于检查原域保持情况。", "",
             f"基准 manifest SHA-256：`{protocol['suite_manifest_sha256']}`。", "",
             "## 训练结果", "",
             "| 组别 | 初始 val UV EPE% | 最佳 val UV EPE% | 最佳轮 | 步数 | 耗时 min |",
             "|---|---:|---:|---:|---:|---:|"]
    bundle = {"protocol": protocol, "training": {}, "evaluation": {}, "paired": {}, "retention": {}}
    for name in ("integrity_audit", "runtime", "split_audit"):
        path = args.experiment / (name + ".json")
        if path.exists():
            bundle[name] = json.loads(path.read_text())
    for name in protocol["arms"]:
        record = json.loads((args.experiment / name / "best.report.json").read_text())
        record["checkpoint_sha256"] = hashlib.sha256((args.experiment / name / "best.pt").read_bytes()).hexdigest()
        bundle["training"][name] = record
        lines.append(f"| {name} | {fmt(record['initial_validation']['coord_epe_pct'])} | "
                     f"{fmt(record['best_score'])} | {record['best_epoch']} | {record['steps']} | {record['elapsed_s']/60:.1f} |")
    lines += ["", "耗时为 wall time，包含运行中的长间隔，不用于比较两组计算性能。", ""]
    for split in ("val", "test"):
        dest = args.experiment / ("comparison_" + split)
        rows = json.loads((dest / "per_sample.json").read_text())
        summary = json.loads((dest / "summary.json").read_text())
        bundle["evaluation"][split] = summary
        failures = [r for r in rows if r["status"] != "ok"]
        lines += ["", f"## {'开发验证' if split == 'val' else '保留测试'}集（90 张）", "",
                  f"共 {len(rows)} 条方法记录，运行异常 {len(failures)} 条。误差与覆盖率均取逐页中位数；缺失指标留空。",
                  "own AD 与覆盖率一起阅读；common 结构指标使用传统方案与三个网络均有效的交集。", "",
                  "| 谱种 | 方法 | own AD% | own 覆盖率% | common 谱线平直度 | common 简谱基线 | 谱种正确率% |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for kind in ("staff", "jianpu", "mixed"):
            for method in methods:
                entry = next(r for r in summary if r["kind"] == kind and r["method"] == method)
                music = entry["common_music"]
                lines.append(f"| {kind} | {method} | {fmt(entry['ad_pct'])} | {fmt(entry['coverage'],100)} | "
                             f"{fmt(music.get('staff_straight'))} | {fmt(music.get('jp_baseline'))} | {fmt(entry['type_accuracy'],100)} |")
        lines += ["", "谱表检测按所有三种谱种的实例汇总（简谱页上误检出的谱表同样计入假阳性）：", "",
                  "| 方法 | 匹配/预测/真值 | precision% | recall% |", "|---|---:|---:|---:|"]
        for method in methods:
            group = [r for r in summary if r["method"] == method]
            matched, predicted, truth = [sum(r["detection_counts"][key] for r in group)
                                          for key in ("staff_matched", "staff_pred_count", "staff_gt_count")]
            lines.append(f"| {method} | {matched}/{predicted}/{truth} | "
                         f"{fmt(matched/predicted if predicted else None,100)} | {fmt(matched/truth if truth else None,100)} |")
        lines += ["", "逐页配对差值（前者 − 后者；负值为改善）。每对方法使用自己的共同有效点，避免传统方案的低覆盖率掩盖网络之间的差异。", "",
                  "| 谱种 | 对比 | ΔAD% 中位数 | AD 改善/有效页 | Δ谱线平直度 | Δ简谱基线 |",
                  "|---|---|---:|---:|---:|---:|"]
        bundle["paired"][split] = []
        for kind in ("staff", "jianpu", "mixed"):
            for left, right in (("data_only", baseline), ("structure_consistency", "data_only")):
                record = paired(rows, left, right, kind)
                bundle["paired"][split].append({"kind": kind, "left": left, "right": right, **record})
                ad = record["ad_pct"]
                lines.append(f"| {kind} | {left} − {right} | {fmt(ad['median_delta'],digits=4)} | {ad['improved']}/{ad['n']} | "
                             f"{fmt(record['staff_straight']['median_delta'])} | {fmt(record['jp_baseline']['median_delta'])} |")
        lines += ["", "按条件分层的几何误差（逐页 own AD% 中位数，每格 18 张，仍需结合完整报告中的覆盖率）：", "",
                  "| 条件 | 传统 | 原模型 | data_only | structure_consistency |", "|---|---:|---:|---:|---:|"]
        for profile in manifest["profiles"]:
            values = [median([r.get("ad_pct") for r in rows if r["profile"] == profile and r["method"] == method])
                      for method in methods]
            lines.append(f"| {profile} | " + " | ".join(fmt(v) for v in values) + " |")
        if failures:
            lines += ["", *[f"- 失败：{r['sample']} / {r['method']}: {r['error']}" for r in failures]]
    lines += ["", "## DeepScores 原域保持（172 张 val）", "",
              "这里使用反演前的网络坐标场，与本表原模型比较；与上面实际输出映射的 AD 不直接横比。覆盖率为均值，误差为中位数。", "",
              "| 模型 | AD% | 局部畸变% | 覆盖率% | 方向错误率% 均值 |", "|---|---:|---:|---:|---:|"]
    for method in methods[1:]:
        entry = json.loads((args.experiment / "retention_val" / method / "summary.json").read_text())["network"]
        bundle["retention"][method] = entry
        lines.append(f"| {method} | {fmt(entry['ad_pct_median'])} | {fmt(entry['local_distortion_pct_median'])} | "
                     f"{fmt(entry['coverage_mean'],100)} | {fmt(entry['orientation_error_rate_mean'],100)} |")
    def test_entry(kind, method):
        return next(r for r in bundle["evaluation"]["test"] if r["kind"] == kind and r["method"] == method)
    jp_old, jp_data, jp_cons = [test_entry("jianpu", m) for m in methods[1:]]
    mx_old, mx_data, mx_cons = [test_entry("mixed", m) for m in methods[1:]]
    outcome = ["## 本轮结论", "",
               f"- 主要收益来自补充训练数据：保留测试中，简谱 own AD 中位数 {fmt(jp_old['ad_pct'])}% → {fmt(jp_data['ad_pct'])}%，混排 {fmt(mx_old['ad_pct'])}% → {fmt(mx_data['ad_pct'])}%。各网络网格覆盖率中位数约 98.5%–98.7%；逐页共同点比较同样支持改善，详见配对表。",
               f"- 结构一致性额外收益较小：简谱 own 基线误差 {fmt(jp_data['music']['jp_baseline'])} → {fmt(jp_cons['music']['jp_baseline'])}，混排 own 谱线误差 {fmt(mx_data['music']['staff_straight'])} → {fmt(mx_cons['music']['staff_straight'])}；两组验证 UV EPE 几乎持平。",
               f"- 未形成全面更优的替代：结构约束组混排谱种正确率 {fmt(mx_cons['type_accuracy'],100)}%，低于数据组 {fmt(mx_data['type_accuracy'],100)}%；DeepScores 原域 AD 从 {fmt(bundle['retention'][baseline]['ad_pct_median'])}% 上升至两组约 {fmt(bundle['retention']['data_only']['ad_pct_median'])}%。",
               "- 两组权重均保留，暂不切换通用默认权重。下一步应补训结构/版面/谱种辅助头，并在验证集选模时同时加入原域保持要求；之后用新的保留素材验证，不继续用本轮 test 调参。", ""]
    position = lines.index("## 固定数据与实验协议")
    lines[position:position] = outcome
    lines += ["", "## 解释范围与复现", "",
              "权重保存在本地实验目录的 `data_only/best.pt` 与 `structure_consistency/best.pt`；原始全量权重没有覆盖。训练配置、历史、曲线、完整评测与数据清单均保留。",
              "单 seed、固定 24 轮的受控实验；不能据此声称不同随机种子上收益稳定。结构一致性约束只包含水平/竖直实例方差，不包含完整的五线等距、字高或歌词对应损失。",
              "投影筛选只作用于本轮新合成素材；DeepScores 回放沿用既有 prepared 数据。完整性复核确认原始权重、15 个训练/评测源码文件及 1,920 个新素材文件的校验和未变，两组的 8 个辅助头张量均与初始化相同。",
              "各档并非互斥的困难因素：标准与强形变档仍可能随机出现阴影、遮挡或模糊。生成投影通过采样方向检查，但预测仍可能折叠，需结合方向错误率、往返检查及覆盖率判断。",
              "首版 train/val 的投影审计发现 9 张局部翻折，已停止初步训练并保留诊断输出；本报告只采用加生成时投影过滤的 v2。修正发生在查看保留 test 的模型表现之前，不依据 test 调参。",
              "谱种正确率与谱表实例精度不等价。完整的五线谱/简谱指标、谱表匹配数、失败页、条件分层结果和配对支持点结果见本地 experiment 输出。", "",
              "```bash",
              ".venv/bin/python scripts/prepare_structure_suite.py --out data/prepared/structure_suite_20261115_v2",
              ".venv/bin/python scripts/check_external_data.py data/prepared/structure_suite_20261115_v2",
              ".venv/bin/python scripts/run_structure_ablation.py --device mps",
              "```", "",
              "生成脚本要求新的空目录；已完成的实验可由运行脚本复用完成标记，未完成的权重或评测目录不会被静默覆盖。",
              f"本地完整结果：`{args.experiment.relative_to(root)}`。"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    image_path = args.out.with_suffix(".preview.jpg")
    bundle["preview_selection"] = preview(args.experiment, suite, baseline,
        json.loads((args.experiment / "comparison_test/per_sample.json").read_text()), image_path)
    lines += ["", "## 困难页对照", "",
              "每种谱种选原模型 AD 最高的测试页，不按新模型改善程度挑选。左到右为平整参考、输入、传统方案、原模型、只增加数据、增加结构一致性。", "",
              f"![困难页对照]({image_path.name})"]
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(bundle, indent=2))
    # Keep the exact generated benchmark manifest in versioned reports too.
    args.out.with_name("structure_suite_20261115_manifest.json").write_bytes((suite / "manifest.json").read_bytes())
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
