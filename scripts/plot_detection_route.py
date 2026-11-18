"""Build static comparison figures from the detection-route experiment."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from test_detection_route import SPECS, map_pass, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = list(map(json.loads, (args.run / "rows.jsonl").open()))
    _, selected = report(rows, list(SPECS))
    all_rows = rows + selected
    columns = ["sample", "method", "kind", "tier", "ad_c", "ld_c", "cover_c", "cover",
               "staff_straight", "vtilt", "jp_baseline", "edge_deg", "kst",
               "folded_cell_fraction", "anisotropy_p99", "local_map_pass", "chosen_method"]
    with (args.out / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in sorted(all_rows, key=lambda r: (r["sample"], r["method"])):
            data = {k: row.get(k, "") for k in columns}
            quality = row.get("notes", {}).get("mapping_quality", {})
            data.update({k: quality.get(k, "") for k in ("folded_cell_fraction", "anisotropy_p99")})
            data["local_map_pass"] = map_pass(row)
            writer.writerow(data)
    by = {(r["sample"], r["method"]): r for r in all_rows}
    names = ["classical", "homr", "stem_connected", "homr_rows", "homr_noends", "homr_map_guard"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in names:
        values = sorted(r["ad_c"] for r in all_rows if r["method"] == name and r.get("ad_c") is not None)
        ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", label=name)
    ax.set(xlabel="Content AD (% page width, lower is better)", ylabel="Fraction of dev pages",
           title=f"Detection variants: {len({r['sample'] for r in rows})} dev pages",
           ylim=(0, 1.02), xlim=(0, None))
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "ad_cdf.png", dpi=160)
    plt.close(fig)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
    except OSError:
        font = ImageFont.load_default(size=15)
    cols = ["input", "classical", "homr", "stem_connected", "homr_rows", "homr_noends"]
    samples = sorted({r["sample"] for r in rows if (args.run / "images" / r["sample"]).exists()})
    for sample in samples:
        width, height, header = 400, 590, 72
        canvas = Image.new("RGB", (width * len(cols), height + header), "#dddddd")
        draw = ImageDraw.Draw(canvas)
        for i, name in enumerate(cols):
            path = (Path("data/bench/doc3d_score") / sample / "photo.jpg" if name == "input"
                    else args.run / "images" / sample / f"{SPECS[name]}.jpg")
            img = Image.open(path).convert("RGB")
            img.thumbnail((width-8, height-8))
            canvas.paste(img, (i*width+(width-img.width)//2, header+(height-img.height)//2))
            draw.text((i*width+8, 5), name, font=font, fill="black")
            if name != "input":
                row = by[sample, name]
                quality = row["notes"]["mapping_quality"]
                draw.text((i*width+8, 27), f"AD {row['ad_c']:.3f}, LD {row['ld_c']:.2f}", font=font, fill="black")
                draw.text((i*width+8, 49), f"fold cells {100*quality['folded_cell_fraction']:.4g}%", font=font, fill="black")
        canvas.save(args.out / f"{Path(sample).name}.jpg", quality=88)


if __name__ == "__main__":
    main()
