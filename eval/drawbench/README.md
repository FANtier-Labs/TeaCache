# DrawBench evaluation

移植自本仓库 `TaylorSeer/eval/drawbench`；内置相同的 200 条 DrawBench 提示词，保留 category 等原始字段。所有命令在仓库根目录运行。安装：`pip install -e '.[eval]'`（或 `uv sync --extra eval`）。生成需要 FLUX.1-dev 权重、访问权限及足够显存；评分首次运行需要下载 OpenAI CLIP ViT-B/32 和 LPIPS AlexNet 权重。

## 完整流程

```bash
bash scripts/flux/run_base.sh
bash scripts/flux/run_teacache.sh
# 多阈值对比，共用同一份基线
THRESHOLDS="0.25 0.4 0.6" bash scripts/flux/run_teacache.sh
```

`run_base.sh` 生成基线；`run_teacache.sh` 依次生成 TeaCache 图像、评分、导出报告。可通过 `MODEL`、`OUTPUT`、`PYTHON`、`NPROC_PER_NODE` 环境变量配置。多 GPU 时每个进程加载完整模型，按提示词分片，无需跨 GPU 共享缓存：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash scripts/flux/run_base.sh --limit 4 --steps 20
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 THRESHOLDS="0.6" bash scripts/flux/run_teacache.sh --limit 4 --steps 20
```

基线与候选必须使用相同模型、数据、seed、steps、guidance、尺寸、samples、limit、warmup-runs 和工作进程数。默认每条提示词 1 张，50 步、1024×1024、guidance 3.5、seed 42，每个进程额外执行 1 次不计分的完整预热生成。`--warmup-runs 0` 可禁用预热。这里的预热不是算法的缓存步数。使用 CPU generator，与 playground 一致；各样本种子为 `seed + sample_id`。

## 计时和指标

生成与 `playground/flux.py` 共用 `teacache.timing.timed_inference`：在 patch 上下文内部创建 CUDA Events，依次执行 start.record、CUDA synchronize、完整 pipeline 调用、end.record、CUDA synchronize，最后 elapsed_time 毫秒转秒。包括文本编码、去噪、VAE 解码和 pipeline 图像后处理对应的事件时间范围；不包括权重加载、patch 安装/恢复、图像磁盘保存和评分。不用脚本墙钟时间作 speedup。

- `baseline_seconds` / `teacache_seconds`：逐图 CUDA Event 耗时；汇总时为算术平均秒数。
- `speedup`：逐图为 baseline_seconds / teacache_seconds；总体和类别为各自配对样本的基线耗时之和 / TeaCache 耗时之和，而非逐图倍率的平均值。大于 1 表示加速。
- `baseline_total_seconds` / `teacache_total_seconds`：总体或类别的样本耗时之和，不是多 GPU 作业的墙钟时长。
- `baseline_clip_score` / `teacache_clip_score`：沿用 TaylorSeer 的 OpenAI CLIP ViT-B/32 图文原始余弦相似度；不是乘 100 或截断后的 CLIPScore 变体。长提示词截断。
- `psnr`：RGB [0,1] 图像相对基线计算；完全相同的图像记为字符串 `"inf"`，另提供有限样本平均值和无穷值计数。
- `ssim`：使用 TorchMetrics 计算 RGB [0,1] 图像相对基线的结构相似度，越高越接近基线，相同图像的理论值为 1（浮点计算可能有微小误差）。采用 11×11 高斯窗口、sigma=1.5、k1=0.01、k2=0.03；逐图计算后，总体和分类取算术平均值。
- `lpips`：AlexNet v0.1 感知距离，越低越接近基线。

相同硬件、驱动、负载下比较。配置记录 CUDA GPU 名称、torch/diffusers 版本及计时协议；不兼容的运行不能配对。画质指标包括 CLIP、PSNR、SSIM、LPIPS。

## 文件、续跑与独立评分

默认输出在 `outputs/drawbench/flux/`：

- `baseline/`：基线图像、config.json 和逐样本 records。
- `threshold_0.6/images/`：候选图像及耗时记录。
- `threshold_0.6/scores/`：逐样本画质、耗时与 speedup。
- `threshold_0.6/report/`：summary.json、summary.csv（总体/分类）、samples.csv（逐图）。

生成跳过已有图像和记录，配置变更需要新输出目录。评分校验样本完整性、配对种子/提示词、图像哈希及耗时；已有评分的输入变化会报错，避免使用过期 speedup。性能重新测量请用新的 OUTPUT，不要复用旧耗时。添加 SSIM 前的评分配置不兼容，请为评分指定新的 `--output` 目录；可复用已生成的基线和候选图像，无需重新生成。

```bash
python -m eval.drawbench.score --baseline outputs/drawbench/flux/baseline \
  --candidate outputs/drawbench/flux/threshold_0.6/images \
  --output outputs/drawbench/flux/threshold_0.6/scores
python -m eval.drawbench.export --scores outputs/drawbench/flux/threshold_0.6/scores \
  --output outputs/drawbench/flux/threshold_0.6/report
python -m unittest discover -s tests -v
```
