"""Dewarp score photos.

    python scripts/dewarp.py photo1.jpg [photo2.jpg ...] --out outputs/run
Writes per input:
    <name>_rectified.png   rectified colour image
    <name>_enhanced.png    reading copy: shading removed, paper white, off-sheet background white
    <name>_binary.png      black/white version of it (with --binary)
    <name>_debug.jpg       detected structures on the rectified image
    <name>_result.json     page type, regions, staff lines, jianpu rows, timings
    <name>_bmap.npy        backward map (output grid stride 4 -> input pixels)
    <name>_page.png        reading copy fitted to the chosen paper (default A4)
    <name>_page_color.png  colour version on the same paper
    <name>_page.pdf        printable reading copy (unless --no-pdf)
    <name>_preview.jpg     small paper-layout preview

Methods
    metric       (default) page homography -> camera + developable page surface
                 fitted to staff lines / jianpu rows / vertical strokes -> residual field
    traditional  legacy A+B+C (polynomial surface), kept for comparison only
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scoredewarp.core.metric_pipeline import MODEL_PRESETS, MetricConfig, dewarp_metric  # noqa: E402
from scoredewarp.core.pipeline import Config, dewarp, result_json  # noqa: E402
from scoredewarp.core.preprocess import clean_page, paper_region  # noqa: E402
from scoredewarp.core.structures import draw_structures  # noqa: E402
from scoredewarp.core.transforms import mapping_quality  # noqa: E402
from scoredewarp.core.document import DocumentConfig, DocumentLayout, save_document  # noqa: E402
from scoredewarp.core.image_input import read_photo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--method", choices=("metric", "traditional"), default="metric")
    ap.add_argument("--no-field", action="store_true", help="skip the residual field after the surface fit")
    ap.add_argument("--output-res", choices=("native", "canvas"), default="native",
                    help="metric: native = rectified output at the photo's resolution (default); "
                         "canvas = the ~1600 px working canvas")
    ap.add_argument("--seg", choices=("off", "homr"), default="off",
                    help="metric: learned staff-line / stem segmentation (needs onnxruntime and "
                         "scripts/download_models.py)")
    ap.add_argument("--models", choices=tuple(MODEL_PRESETS), default="off",
                    help="metric: small detection models (geometry stays the solver): page = U2-Netp sheet "
                         "segmentation (sheet mask, page quad, surface selection by the sheet outline); "
                         "all = page + homr staff lines. Weights in models/")
    ap.add_argument("--no-rotate", action="store_true",
                    help="metric: do not turn pages whose staves run vertically (photo taken sideways)")
    ap.add_argument("--crop-content", action="store_true",
                    help="metric: output only the detected content box instead of the whole page")
    ap.add_argument("--binary", action="store_true", help="also write a black/white reading copy")
    ap.add_argument("--paper", choices=("a4", "a3", "letter", "custom", "adaptive"), default="a4",
                    help="output paper; adaptive preserves the rectified canvas")
    ap.add_argument("--orientation", choices=("portrait", "landscape"), default="portrait")
    ap.add_argument("--size-mm", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"),
                    help="custom paper dimensions; orientation determines their order")
    ap.add_argument("--margin-mm", type=float, default=10.0, help="minimum paper margin (ignored for adaptive)")
    ap.add_argument("--dpi", type=int, default=300, help="export resolution, default 300")
    ap.add_argument("--no-pdf", action="store_true", help="export images only")
    args = ap.parse_args()
    try:
        document = DocumentConfig(args.paper, args.orientation, args.dpi, args.margin_mm, args.size_mm)
        if document.paper != "adaptive":
            DocumentLayout.fit((1, 1), document)  # validate rounded printable area before processing
    except ValueError as e:
        ap.error(str(e))
    os.makedirs(args.out, exist_ok=True)
    for path in args.inputs:
        try:
            img = read_photo(path)
        except ValueError as e:
            print(f"cannot read {path}: {e}", file=sys.stderr)
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        if os.path.basename(path) == "photo.jpg":  # benchmark / synthetic folders
            name = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if args.method == "metric":
            preset = dict(MODEL_PRESETS[args.models])
            if args.seg != "off":
                preset["seg"] = args.seg
            cfg = MetricConfig(**{**preset, "residual": "none" if args.no_field else MetricConfig.residual,
                                  "full_page": not args.crop_content, "output_res": args.output_res,
                                  "rot90": not args.no_rotate})
            res = dewarp_metric(img, cfg)
        else:
            res = dewarp(img, Config(use_field=not args.no_field))
        res.stages["method"] = args.method
        res.stages["mapping_quality"] = mapping_quality(res.bmap, (img.shape[1], img.shape[0]))
        base = os.path.join(args.out, name)
        cv2.imwrite(base + "_rectified.png", res.image)
        sp = getattr(res.structures, "spacing_est", None)
        paper = paper_region(res.paper, res.image, spacing=sp, content=getattr(res, "content", None))
        reading = clean_page(res.image, paper, sp)
        cv2.imwrite(base + "_enhanced.png", reading)
        binary = None
        if args.binary:
            binary = clean_page(res.image, paper, sp, mode="binary")
            cv2.imwrite(base + "_binary.png", binary)
        layout = DocumentLayout.fit(res.bmap.out_size, document)
        page_color = layout.render_photo(img, res.bmap)
        page_paper = layout.render_mask(paper)
        page_spacing = sp * layout.matrix[0, 0] if sp is not None else None
        page_reading = clean_page(page_color, page_paper, page_spacing)
        page_binary = clean_page(page_color, page_paper, page_spacing, mode="binary") if args.binary else None
        save_document(base, page_reading, page_color, layout, binary=page_binary, pdf=not args.no_pdf)
        res.stages["document"] = layout.metadata()
        cv2.imwrite(base + "_debug.jpg", draw_structures(res.image, res.structures))
        np.save(base + "_bmap.npy", res.bmap.grid.astype(np.float32))
        with open(base + "_result.json", "w") as f:
            json.dump(result_json(res), f, ensure_ascii=False, indent=1, default=float)
        print(f"{path}: type={res.page_type} rectified={res.bmap.out_size} "
              f"paper={args.paper} size={layout.out_size} correction={res.timings['total']:.1f}s")


if __name__ == "__main__":
    main()
