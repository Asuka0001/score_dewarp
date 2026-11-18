# homr staff segmentation model

`segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx` is the segmentation
network of [homr](https://github.com/liebharc/homr) (UNet adapted from
[oemer](https://github.com/BreezeWhite/oemer)), unchanged, downloaded from
<https://github.com/liebharc/homr/releases/download/onnx_checkpoints/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.zip>.

- License: AGPL-3.0, as published by homr (code and weights; see homr's
  `LICENSE` and `LICENSE_FAQ.md`).
- Training data: DeepScoresV2 (CC BY 4.0).
- Input: (N, 3, 320, 320) float32, grey page replicated to 3 channels, 0–255.
  Output: (N, 6, 320, 320) logits — 0 background, 1 stems / rests / barlines,
  2 noteheads, 3 clefs / key signatures, 4 staff lines, 5 other symbols.
- sha256 `6ed36640db4ef5d223098b6d5efe4eda97c66b24a2c72faab8a018c749003a8d`

Used by `scoredewarp/core/learned.py` (`--seg homr`). Included here for a
personal, non-commercial demo.
