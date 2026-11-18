# score_dewarp — 乐谱照片平直化校正（五线谱 / 简谱 / 混排）

输入一张手机拍摄的乐谱照片（透视、卷曲、褶皱、阴影），输出：

- **校正后的彩色图**（`_rectified.png`）和**阅读副本**（`_enhanced.png`：去阴影、纸面置白、纸外置白；`--binary` 另存黑白版），分辨率与输入照片一致（见下）
- **谱种标签**：`staff` / `jianpu` / `mixed` / `unknown`
- **结构**：谱线折线（含线号）、简谱数字框（含 0–7 识别）、文字行框、版面区域（`_result.json`）
- **变换场**：输出像素 → 原图像素的 backward map（`_bmap.npy`）
- **可打印纸页**：默认 A4 竖版、300 DPI、至少 10 mm 白边，完整等比例居中排版；导出 `_page.png`（阅读版）、`_page_color.png`（彩色版）、`_page.pdf` 和 `_preview.jpg`。

不需要训练数据：页面形变由检测到的乐谱结构（谱线、简谱数字行、竖线、谱表端点）反求，全部是几何优化。可选接入一个现成的轻量五线谱分割网络（`--seg homr`），只负责检测谱线和符干，不参与求形变（见“可选：学习检测”）。

## 快速开始

```bash
cd score_dewarp
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 校正照片（可一次传多张）
.venv/bin/python scripts/dewarp.py my_score.jpg --out outputs/run --binary
```

输入支持 JPEG、PNG、WebP、BMP、TIFF、HEIC 和 HEIF，也会按照片的 EXIF 方向旋正。macOS 上 HEIC/HEIF 由系统的 `sips` 解码为临时无损 PNG 并验证文件可读，原照片不变；其他系统如 Pillow 无 HEIF 解码插件，需要先自行转换成 JPEG 或 PNG。

### 选择输出纸张

```bash
# 默认：A4 竖版，300 DPI，10 mm 白边，同时导出图片与单页 PDF
.venv/bin/python scripts/dewarp.py my_score.jpg --out outputs/a4

# A3 横版、15 mm 白边
.venv/bin/python scripts/dewarp.py my_score.jpg --paper a3 --orientation landscape --margin-mm 15

# 自定义纸张 190 × 260 mm；横竖方向由 --orientation 决定
.venv/bin/python scripts/dewarp.py my_score.jpg --paper custom --size-mm 190 260

# 自适应：保留矫正画布尺寸和已有白边，不强制标准纸型；仅导出图片
.venv/bin/python scripts/dewarp.py my_score.jpg --paper adaptive --no-pdf
```

`--paper` 支持 `a4`、`a3`、`letter`、`custom`、`adaptive`，`--dpi` 设置导出分辨率。标准纸型完整容纳矫正画布，不裁切、不分别拉伸横纵比例；画布比例与纸型不同时，部分边缘留白会大于设定值。横版改变纸张方向，不旋转谱面。A4 在 300 DPI 下为 2480 × 3508 像素；PDF 物理尺寸由像素与 DPI 决定，有不足一个像素的取整误差。放大不会增加原照片细节。

`_preview.jpg` 可直接查看排版效果；`_page.pdf` 默认使用阅读副本，一张照片对应一个 PDF。`--binary` 另外导出纸张尺寸的 `_page_binary.png`。纸页彩色版从原照片按最终纸张像素一次采样，阅读版再在该分辨率清理，避免把粗检测用的约 1600 像素图放大；照片本身不清晰时仍不能凭空补出细节。选择纸型只控制最终排版，不假定被拍摄的原纸也是该规格，也不修复上游的页面分离或严重褶皱问题。

**输出分辨率**：检测和拟合在约 1600 像素宽的工作画布上进行；最后把变换按"一个输出像素约对应一个照片像素"缩放（页面内容区局部尺度的中位数，`stages.output_scale`），再从原照片一次采样得到 `_rectified.png` / `_enhanced.png`，结构坐标同步缩放。例：3024 × 4032 的 CPMS 照片输出 3001 × 3970（旧版为 1591 × 2105）。透视使页面各处采样密度不同，离镜头近处略降采样、远处略升采样。`--output-res canvas` 保留旧的画布输出；基准测试固定用画布（几何指标经映射计算，与分辨率无关）。标准纸型导出仍按 `--dpi` 采样：A4 300 DPI 宽 2480 像素，低于 12 MP 照片的原生分辨率，需要保留全部细节时用 `--paper adaptive` 或提高 `--dpi`。

`_rectified.png`、`_enhanced.png`、`_debug.jpg`、`_bmap.npy` 和结构坐标对应同一张输出图。`_result.json` 的 `stages.document` 记录纸张大小、DPI、像素尺寸和 `rectified_to_document` 矩阵，供将结构坐标映射到纸页；`_bmap.npy` 不能直接当作 `_page.png` 的坐标场。

其他选项：`--no-field` 只用曲面模型、不跑残差场；`--crop-content` 只输出内容框；`--method traditional` 是旧 ABC 方案，仅作对照。单张 12 MP 照片约 10 秒（Apple Silicon，单进程）。

## 方法

```
照片 ─► A 页面单应 + 纠斜 ─► 结构检测 ─► B 针孔相机 + 可展柱面拟合 ─► 重新检测 ─► C 残差场 ─► 输出范围 / 页边距 / 阅读副本
```

- **A**（`core/page.py`）：检测页面四边形做透视矫正，再用投影轮廓纠斜；检测不到四边形时只纠斜。
- **结构检测**（`core/structures.py`）：游程统计估计线粗和线距，逐列跟踪谱线并聚成五线谱表；简谱按字高把连通域链接成行，用模板识别 0–7；竖线（小节线、符干）只取细的部分，去掉符头 / 符尾连接处，否则符干整体偏约 1°。
- **B**（`core/gcs.py`）：纸张弯曲但不拉伸，页面用一般柱面（可展曲面）表示，加针孔相机。未知量是相机位姿、母线方向、截面斜率曲线和每个结构的平面坐标；约束是重投影误差加各谱种的几何关系（见下表）。曲面是等距展开的，所以输出的比例是度量正确的，不需要再估计长宽比。
- **C**（`core/field.py`）：在 B 的输出上重新检测，解一个网格上的前向场 `(U, V)`。所有先验都写成关于 U 或 V 的线性约束：同一谱线 V 相等、相邻谱线差一个线距、竖线 U 相等、各行谱表的左右端 U 相等（页边对齐），加平滑项，用 IRLS 求稀疏最小二乘。全局比例不变（`keep_scale`）。
- **输出**（`core/metric_pipeline.py`）：纸张边缘（GrabCut 掩码 + 纸面直线检验）限定范围，范围再扩到装下所有印刷内容，四周补 3% 白边；阅读副本只保留纸面。各阶段都表示成 `BackwardMap`，只对原图重采样一次。

| | 五线谱 | 简谱 |
|---|---|---|
| 主要线索 | 贯穿全宽、5 线等距的长细线；小节线、符干；谱表左右端对齐 | 没有长线：数字 0–7 字高一致（顶线 + 底线双约束）；小节线、数字和汉字的竖笔画 |
| B 中的先验 | 每个谱表一组 `Y0, s`：第 i 线 `Y = Y0 + i·s`，各谱表的 `s` 软约束一致；各行谱表的左右端共用一个 X | 每行一组 `Y_top, h`：数字顶点在 `Y_top`、底点在 `Y_top + h`；各行 `h` 软约束一致 |
| C 中的先验 | V：沿线相等、相邻线差 = 线距；U：沿线弧长按局部线距归一；谱表端点 U 相等 | V：顶线、底线各自相等，底 − 顶 = 字高；U：相邻字间距按局部字高归一 |

混排页面上两类约束同时作用于同一个曲面和残差场。

### 可选：学习检测（`--seg homr`，默认关闭）

参考扫描软件 / 文档矫正的常见做法（轻量模型只做检测，几何求解器负责展平，如 RDGR、PaddleOCR 文档预处理），接入了开源 OMR 项目 [homr](https://github.com/liebharc/homr) 的五线谱分割网络（UNet，DeepScoresV2 训练，ONNX，57 MB；输出谱线、符干 / 小节线、符头、谱号、其他符号）。

```bash
.venv/bin/pip install onnxruntime
.venv/bin/python scripts/download_models.py          # 权重下载到 data/external/models/homr/，不随仓库分发
.venv/bin/python scripts/dewarp.py my_score.jpg --seg homr
```

- **推理**（`core/learned.py`）：自写滑窗推理，复现 homr 的预处理（灰度 + CLAHE、320 px 窗口、半窗步长）；输入宽度按实测线距缩放到约 16 px / 线距（homr 固定宽 1920，在低分辨率照片上会整排漏检谱表）。macOS 上用 CoreML，约 2–5 s / 页；`SCOREDEWARP_SEG_CACHE` 可缓存结果。分割图在照片上算一次，随图像用同一个 `BackwardMap` 重采样，每轮重新检测都复用。
- **融合**（`structures.py`）：谱表取网络谱线掩码与形态学检测中覆盖谱线总长更长的一方；谱表区内的竖笔画取网络的符干 / 小节线（不含符头，从源头去掉符干约 1° 的偏斜），并把谱线像素补回（网络类别互斥，符干在每条谱线处被切断）；网络找不到谱表时（简谱页）完全走原流程，结果逐像素不变。
- **效果**：作为检测器明显更好，但对展平本身基本中性，因此未设为默认：

| 评测 | 手工检测 | + homr |
|---|---:|---:|
| dev 16 页，谱线召回 / 位置误差（线距） | 0.875 / 0.031 | **0.917 / 0.029** |
| dev 16 页，竖笔画偏斜 | −0.64° | **−0.51°** |
| dev 120 页端到端，内容区 AD 中位（页宽 %） | 0.875 | 0.875（28 页好 / 22 页差） |
| CPMS 60 张照片，照片上检测：符头音高位置准确率 | 0.726 | **0.868**（50 张好 / 2 张差） |
| CPMS 端到端：同一裁判读校正结果的符头位置准确率 | 0.879 | 0.873 |
| CPMS 端到端：校正后谱线残余弯曲（线距）/ 倾斜 | 0.020 / 0.04° | **0.016 / 0.02°** |

CPMS 上同一张校正图，手工检测器读出的符头位置准确率为 0.76，网络读出 0.88：网络的价值主要在输出结构（谱线、音高位置，便于 OMR），而非形变求解。homr 的代码和权重均为 AGPL-3.0（训练数据 DeepScoresV2 为 CC BY 4.0），本仓库只在运行时下载权重。

## 评测

基准用 Doc3D 的真实纸张形变（UV 图）给乐谱页重新贴图（`data/bench/doc3d_score`，dev / test 各 120 页，五线谱、简谱、混排，轻 / 中 / 重三档），所有指标都来自真值，不经过我们的检测器。

```bash
.venv/bin/python scripts/download_datasets.py deepscores   # 以及 doc3d；CPMS 真实照片用 download_cpms.py
.venv/bin/python scripts/make_doc3d_bench.py --out data/bench/doc3d_score --per-split 120
.venv/bin/python scripts/bench.py --data data/bench/doc3d_score/dev --methods input abc metric --out outputs/bench/dev
.venv/bin/python scripts/bench.py --data data/bench/doc3d_score/dev --methods metric "gcs:c.ends_w=4" --out outputs/bench/x   # 参数变体
.venv/bin/python scripts/make_compare_figures.py --out reports/compare_xxx                                                  # 对比图
```

dev 中位数（2026-11-18，当前指标定义）：

| 指标 | 原图 | 旧 ABC | 主线 |
|---|---:|---:|---:|
| 内容区 AD（页宽 %） | 2.44 | 1.77 | **0.88** |
| 内容区 LD（局部畸变 %） | 13.7 | 11.9 | **6.9** |
| 可见页面保留比例 | 1.000 | 0.881 | 0.998 |
| 谱线平直度（y 标准差 / 线距） | 1.30 | 0.71 | **0.15** |
| 竖线倾角（°） | 2.74 | 2.50 | **0.75** |
| 简谱基线平直度（/ 字高） | 0.51 | 0.24 | **0.084** |
| 页边倾斜（°） | 2.26 | 1.95 | **0.54** |
| 首末行宽度差（%） | 7.6 | 5.0 | **1.6** |

结构指标在整页对齐之后测量（Doc3D 的页面长宽比是任意的，对齐分别拟合 X、Y 尺度），因此一个把页面压窄的方法不能靠此降低角度指标。test 集只在 2026-11-17 评过一次，当时用的是旧指标定义，结果见报告。

### 真实照片：CPMS 符头–谱线关系

CPMS（60 张手机拍摄的印刷五线谱，11092 个符头）标注了每个符头在谱表中的位置（以中线为 0 的半线距级数）。`scripts/eval_cpms.py` 用它直接衡量“符头与谱线关系”：

- 标注坐标在一张缩小过的照片上、比例未给出且每张不同（0.42–0.46）：逐张搜索使标注点落在符头上的比例（命中率 100%），再吸附到符头中心；两种方法共用这套坐标。
- **照片上检测**：每个符头按检测到的五条谱线读出位置，与标注比对；不在任何检测谱表内记为错。
- **端到端**：符头经变换映射到校正结果上，由一个固定的“裁判”检测器（手工或网络，两种方法用同一个）读出位置；同时报告校正后谱线的残余弯曲和倾斜。

```bash
.venv/bin/python scripts/download_cpms.py
SCOREDEWARP_SEG_CACHE=outputs/segcache .venv/bin/python scripts/eval_cpms.py --out outputs/cpms_eval --workers 4
```

结果见上节表格。

## 目录

```
scoredewarp/
  core/   page.py (A)  structures.py (检测)  gcs.py (B)  field.py (C)  metric_pipeline.py (主线)
          learned.py (可选 homr 分割)  pipeline.py (旧 ABC 与共用的收尾)  surface3d.py / line_mesh.py (旧 ABC 用)
          preprocess.py (光照、二值化、纸面、阅读副本)  transforms.py (BackwardMap、原生分辨率缩放)
  eval/   geometry.py  metrics.py  structures.py  real_score.py      评测指标
  synth/  render.py  fonts.py  doc3d_retexture.py                   渲染平整乐谱页、Doc3D 重贴图
scripts/  dewarp.py (CLI)  bench.py  bench_report.py  make_doc3d_bench.py  eval_cpms.py (CPMS 符头–谱线评测)
          make_compare_figures.py  make_route_figures.py  make_align_figures.py
          download_datasets.py  download_cpms.py  download_models.py (homr 权重)
tests/    python -m unittest discover tests
reports/  metric_route_20261117.md（主线报告，补充一至五）与对比图
```

## 已知局限

2026-11-18 稳定性实验：主线默认检查残差网格的局部有效性，遇到翻转或极端伸缩时缩步，仍无效则保留上一轮。120 页 dev 未触发检查、AD 逐页不变，因此尚无实测精度提升。双起点完整优化改善尾部误差但仍有退化页，竖线按跨线数量赋权整体变差，均未设为默认。详见 [稳定性消融报告](reports/stability_20261118.md)。

1. **两页摊开、背景杂乱的照片**（2026-11-18 的手持拍书测试）：主页面能展平，但输出里会留下对面那页、桌面和手，阅读副本把背景变成灰色纹理。需要只保留主页面。
2. **对面那页不会被展平**：曲面模型一次只表示一张纸。
3. **严重褶皱页上曲面拟合会陷入局部解**（如 dev 0004）：输入稍有变化结果就可能差很多，需要多起点拟合。
4. **方向**：90° 检测默认关闭，180° 未处理。
5. **输出范围外推**：个别重度形变页上，曲面稍有变化就会把画布外推到页外并带出重复内容（如 dev 0098 在 `--seg homr` 下 AD 升到 9.6）；与是否使用分割无关，是开关分割对比中尾部波动的主要来源。

## 历史

- 主线的设计、调参和每一轮改动的评测：[reports/metric_route_20261117.md](reports/metric_route_20261117.md)。
- 2026-11-17 之前尝试过的方案（多任务网络 UV、UVDoc、网络 + 优化混合、合成数据训练、辅助头）已放弃，代码于 2026-11-18 从仓库删除，需要时用 git 标签取回：`git checkout archive/pre-cleanup-20261118`；它们的报告在提交 `9acb325` 中（`git checkout 9acb325 -- reports/`）。
