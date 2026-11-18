# U²-Netp salient-object segmentation model

`u2netp.onnx` is the small U²-Net variant (U²-Netp, ~1.1 M parameters) by
Qin et al., "U²-Net: Going Deeper with Nested U-Structure for Salient Object
Detection", Pattern Recognition 2020, <https://github.com/xuebinqin/U-2-Net>,
unchanged, as converted to ONNX by rembg and downloaded from
<https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx>.

- License: Apache-2.0 (U-2-Net and rembg).
- Training data: DUTS-TR (generic salient objects; no document or Doc3D data,
  so it has not seen the Doc3D bench pages).
- Input: (1, 3, 320, 320) float32 RGB, ImageNet mean / std normalised.
  Output: first of seven (1, 1, 320, 320) maps, foreground probability.
- sha256 `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`

Used by `scoredewarp/core/pagenet.py` (`--page-seg u2netp`) to find the sheet
in the photo; it is a detector only and does not take part in the geometry.
