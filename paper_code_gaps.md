# FaceLinkGen: `main.pdf` 与仓库代码的实质性差异

核查方式：8 个重叠 finder agent（各领一个论文章节，代码覆盖面交叉）→ 3 个独立交叉验证 agent（反驳 / 数字追溯 / 覆盖完整性）→ 41 条候选，逐项人工复核取证。

本文件只保留**不能用"代码后来被改过"解释、且会改变已报告结果或其解读**的条目。被过滤掉的条目见文末清单。

---

## 1. Table X 的实现与论文描述的不是同一个实验

论文 p8-p9：*"we appended a multi-head prediction MLP to a CNN and trained it for one epoch on the FairFace dataset [41] to infer soft biometrics, predicting age, gender, and race (7 categories). Evaluation was performed on 10k images."*，p9 并强调 *"a model can learn a direct mapping from the template to these attributes **without facial reconstruction**"*。

实际路径 `evaluation/attributes/face_attributes.ipynb` + `prep_names.ipynb` + `predict.py`：

| 论文 | 代码 |
|---|---|
| 接多头 MLP，训练一个 epoch | 无任何训练。`predict.py:73-77` 加载 FairFace 作者发布的预训练 ResNet-34（`res34_fair_align_multi_7_20190809.pt`），`.eval()`；全文件无 optimizer / loss / backward |
| 直接从 protected template 推断，不经重建 | `face_attributes.ipynb` cell 7 先用 Arc2Face 从 student embedding 生成图片存到 `attributes_<method>/gen`，属性预测跑在这些**生成图**上 |
| 属性 accuracy | `prep_names.ipynb` cell 15 用同一模型对**原图**的预测当 ground truth：`age_gt.append(inp["age"])`，真标签版本 `# age_gt.append(age_label[ds[i]["age"]])` 被注释掉。即报告的是"生成图预测 vs 原图预测"的一致率 |
| 10k images | `att_fracface.csv` / `att_minusface.csv` / `att_partialface.csv` 各 100 gen + 100 in；`att.csv` 500 + 500 |
| Age MAE（年） | `predict.py:120,128` 是 `outputs[9:18]` 的 9 类年龄段 argmax（0-2/3-9/…/70+）；`prep_names.ipynb` cell 17 用 `int(bucket_start) + 5` 转数值后算 MAE |
| Random Baseline 行、Original Image 上界行 | 无实现代码 |

`prep_names.ipynb` cell 17 保留的唯一执行输出（打印标签为 "FracFace"）是 `Age MAE: 6.02 / Race Acc: 0.56 / Gender Acc: 0.77`，Table X 的 FracFace 行是 `4.737 / 0.673 / 0.925`。

---

## 2. Table IX 的 n 不包含用于 early stopping 的 2000 对验证集

Table VIII / IX 的数字来自 `attacks/deid_linkage/results/retrieval_*_{n}_converge.json`（已核对：PerceptFace n=100 的 `top1_hit = 0.61484` / `topK_hit = 0.88645` 精确对应 Table IX 的 0.6148 / 0.8865）。

`_converge` 由 `f2_distill_converge.py` 产生：
- `:42` `val_names = ...open(WORK + '/splits/ffhq_gate_val.txt')`
- `a6_rebuild_splits.py:21,73` `N_GATE = 2000`
- `:61-63` val 侧同样是 (protected, original) 对
- `:116-131` 用 `val_cos` 选 `best.pt`

即 n=100 那一列，攻击者实际可用的 (protected, original) 对是 2100 而不是 100。

与同目录固定 5000 步、不使用验证集的 `retrieval_*_{n}.json` 相比，Top-1 的选择增益：

| 方法 | n=100 | n=200 | n=500 | n=2000 |
|---|---|---|---|---|
| PerceptFace | +0.0587 | +0.0213 | +0.0006 | −0.0013 |
| CanFG | +0.0632 | +0.0619 | +0.0213 | +0.0058 |
| CanFG-Ano | +0.0535 | +0.0697 | +0.0258 | +0.0181 |
| TIP-IM | +0.0219 | +0.0077 | −0.0045 | +0.0013 |

---

## 3. Table II 的跨方法 query/key 不保证是不同的两张图

`notebooks/fracface/linkage2.ipynb` 是 Table II 的来源（cell 32 输出 `0.786288 / 0.753664 / 0.730496 / 0.720567 / 0.802837 / 0.786761`，与表中数字逐一对应）。

- cell 12：`*_embeddings_1 = [v[0][0] for k,v in sorted(...)]`、`*_embeddings_2 = [v[1][0] ...]` — query 取每个身份在**该方法自己 pkl 顺序**下的第 0 张，key 取第 1 张
- cell 6：notebook 自己的检查 `np.all(np.array(minusface_data['filenames']) == np.array(partialface_data['filenames']))` 输出 **`np.False_`**
- cell 7：`set(minusface) - set(partialface)` 与反向都是空集 — 集合相同，顺序不同

所以在跨方法格（如 query=MinusFace、key=PartialFace）里，B 的第 1 张可能就是 A 用作 query 的那张源图，违反 p5 *"The query and key for each pair are generated from different images of the same identity"*。身份对齐本身没问题（两侧都按 ID `sorted`）。

同一 hold-out 里有 392 个身份只有 2 张图，这些身份在跨方法格中同图概率为 50%。

---

## 4. Table II 的 MinusFace 列是在评测 hold-out 上挑出的中间 epoch

`attacks/minusface/insight_train.py:233` `epochs = 10`，`:305` 每个 epoch 都 dump `log/insight_student_embeddings_val_epoch{e}.pkl`；`linkage2.ipynb` cell 0 读的是 `epoch{4}`。挑选依据的 val 集就是 Table II 报告的那 2,115 身份 hold-out。

---

## 5. de-identification 部分的训练目标不是 Eq(2)

Eq(2) 是纯余弦。三个主 PPFR 攻击确实是纯余弦（`attacks/{minusface,partialface,fracface}/insight_train.py` 的 `:267 / :212 / :233` 均写作 `loss = cos# + mae# + trip`，mae 与 trip 被注释掉）。

Table VIII / IX 那一族全部是 `loss = cos + mae + trip`，其中 mae 与 in-batch triplet 各带 10× 权重：

- `attacks/deid_linkage/f1_distill.py:88-95`（`mae = F.l1_loss(s_n, t_n) * 10`，`trip = ... * 10`，`neg = t_n[torch.roll(...)]`）
- `attacks/deid_linkage/f2_distill_converge.py:109`
- `attacks/perceptface/insight_train.py:95`
- `attacks/perceptface/insight_train_lowdata.py:87`
- `attacks/canfg/insight_train.py:198`

triplet 项是对其他身份 teacher embedding 的对比目标，正是 Table VIII/IX 检索指标（Top-1 / Top-0.5% / rank）所测量的判别压力。两组实验的损失不同。

---

## 6. Table VIII / IX 的 Avg rank_all 是按 (query, gallery 图) 对做 micro-average

`attacks/deid_linkage/g1_eval.py`：

```python
:87  rank_all = np.concatenate([np.nonzero(row)[0] + 1 for row in same])   # 扁平成一个向量
:94  'avg_rank_all': float((rank_all / len(gal_rel)).mean())              # 只平均一次
```

同函数里的兄弟指标是 per-query 再平均（`:86 topk_recall = same[:, :K].sum(1) / n_same` → `:93 .mean()`）。

split 规模：`gallery = 10723`、`query = 1550`（`results/retrieval_*.json`），`a6_rebuild_splits.py:48-53` 把某身份所有非 query 图都放进 gallery，1550 个 query 平均拥有 ~4.4 张同身份 gallery 图，而单一高频身份（LFW 里 George_W_Bush 约 530 张）可占其中数百张。拥有 m 张同身份 gallery 图的 query，其 mean rank 有 ≥ (m+1)/2 的硬下界，因此被过度加权的高 m query 无法贡献小的归一化 rank。

---

## 7. U-Net baseline 在仓库自己的 artifact 里并没有"大部分时候失败"

`attacks/channel_experiments/eval_apis.py:65-70` 对 `ours` 与 `unet` 使用完全相同的双条件判定（两张图都检测到脸 **且** `confidence > thresholds["1e-5"]`）和同一分母（`:137` / `:202` 均 assert 300），两侧都是单次尝试。

`artifacts/new_plan/api/{unet,ours}/*/summary.json` 的 `facepp_at_least_one_rate`：

| 设置 | U-Net | Ours |
|---|---|---|
| fracface_fixed | **0.990** | 0.947 |
| partialface_fixed | 0.937 | 0.953 |
| minusface_random | 0.503 | 0.940 |
| fracface_random_train_fixed_test | 0.363 | 0.897 |
| partialface_random_train_fixed_test | 0.550 | 0.930 |

fixed-channel（主实验的 oracle 设置）下 U-Net 平手或反超。论文 p5 *"the U-Net attack failed to generate valid faces most of the time"* 与之矛盾。

相关：U-Net 一律实例化为 `init_features=3`（`train_unet.py:124`、`dump_reconstructions.py:151`、`attacks/unet_reconstruction/` 下所有 notebook），论文引用的 [25] 实现默认为 32。同一个 3-feature 网络训 10 epoch 时是 57.8%（`attacks/unet_reconstruction/RESULTS_fracface_ablation.md:36`），`train_unet.py:137` 训 20 epoch 后变 99.0%。

*注：论文 p5 对 baseline 表的引用是坏的（`Table ??`），无法确定正文用的具体是哪一组数。*

---

## 8. 像素级 SSIM / PSNR 在未对齐的图上计算

`attacks/channel_experiments/measure_reconstruction_metrics.py:76-107` 只把两侧 resize 到 112×112，无检测、无裁剪、无对齐。"ours" 一侧是 512×512 的 Arc2Face 人像，"原图" 是已对齐的 112×112 crop。

结果："ours" 9.82 dB / SSIM 0.141 — 低于常数灰图（11.35 dB）和常数平均脸（SSIM 0.368）；U-Net 侧是 19.0–20.0 dB / SSIM 0.75。p5 的 *"Both methods score poorly on the pixel-level metrics, and ours scores lower than the U-Net baseline"* 里，"ours 更低"由几何未配准决定，"both poorly" 对 U-Net 不成立。

---

## 9. PartialFace 的归一化用了 batch 中第 0 个样本的 min/max

`attacks/partialface/insight_train.py:121`：

```python
final = (final - final.amin(dim=(1,2,3), keepdim=True)[0]) / (final.amax(dim=(1,2,3), keepdim=True)[0] - final.amin(dim=(1,2,3), keepdim=True)[0] + 1e-5)
```

`amin(dim=(1,2,3), keepdim=True)` 已返回 `(B,1,1,1)`，多出的 `[0]` 取第一个样本的标量并广播到整个 batch。`:190` batch_size=256、`shuffle=True`，因此每个 template 每个 epoch 被一个随机的其他样本的 min/max 缩放；val loader 不 shuffle，train 与 val 的缩放方式系统性不同。

同仓库正确写法：`attacks/minusface/insight_train.py:162-164`、`attacks/channel_experiments/train_ours.py:197-207`（均无 `[0]`）。

---

## 10. MinusFace 学生的 train / test 前处理不一致，但加载同一 checkpoint

| | train (`insight_train.py`) | test (`insight_test.py`) |
|---|---|---|
| BGR→RGB | `:125` `img_s = img_s[..., ::-1]` | 无（`:110-115`） |
| per-sample min-max stretch | `:162-167` `minv/maxv = imgs.amin/amax(dim=(1,2,3))` → `(imgs-minv)/(maxv-minv+1e-6)` → `(imgs-0.5)/0.5` | 无（`:120-131` 的 `convert_batch` 只有 `tf_student(imgs)`）|

`out[5]` 来自 `idct_transform`，被 clamp 到 [0,1] 且实际动态范围窄，去掉 stretch 会明显改变输入对比度；通道序交换是额外的系统性偏移。

---

## 11. MinusFace 转换模型的输入尺度错误

- `methods/minusface/minusface.py:90` 注明 `# for DCT: image-form inputs have a range of [-1, 1], and inverse DCT produces [0, 1]`；`:177` 模块自带 demo 是 `x = x * 2 - 1`
- `methods/minusface/train.yaml:26` `RGB_MEAN: [0.5,0.5,0.5]  # for normalize inputs to [-1, 1]`
- `attacks/minusface/insight_train.py:69-71` 的 `tf_conv` 只有 `Resize + ToTensor`（[0,1]），`:156` 直接 `conversion_model(conv_raw)[5]`，无 `*2-1`
- `dct_transform` 第一步是 `x = x * 0.5 + 0.5`（`attacks/partialface/processing_utils.py:19`），所以 [0,1] 输入变成 [0.5, 1.0]：对比度减半 + 大 DC 偏移

stage-1 U-Net generator 在训练分布之外运行，`x_residue = x_up - x_encode_up` 不是 MinusFace 实际发出的残差。

---

## 12. PartialFace 只暴露 6 个通道子集中的 3 个

`attacks/partialface/processing_utils.py:158-163` 用 label 决定样本可选的子集：

```python
label_idx_mod = [int(labels[i]) % len(choice_index) for i in range(b)]
split_idx = [i + b * choice_index[label_idx_mod[i]][idx_within_choice[i]] for i in range(b)]
```

所有调用点都传常量 `[1]*b`：`attacks/partialface/insight_train.py:118`、`insight_test.py:116`、`attacks/channel_experiments/train_ours.py:190`、`dump_reconstructions.py:184`、`attacks/transfer/insight_test_partial.py:205`。`1 % 20 == 1` → 恒为 `choice_index[1]`。54 个高频通道里只有 27 个会出现在任何 template 中（训练和评测都是），学生要学的映射只有 3 个 mode 而非 6 个；测试时 query/gallery 也来自同样受限的 3 个。

---

## 13. CanFG 攻击把检测失败的 teacher target 填成零向量，损失仍朝零回归

`attacks/canfg/extract_embeddings.py:80-85`：

```python
emb = get_embedding(path)
if emb is None:
    failed += 1
    embeddings[path] = np.zeros(512, dtype=np.float32)
```

这些样本进入 `attacks/canfg/insight_train.py:198` 的 `loss = cos + mae + trip`：余弦项 `F.normalize(0) = 0` 贡献常数 1.0 且无梯度，但 10× L1 项变成 `10 * mean|s_n|`，主动把学生的归一化 embedding 推向零；triplet 的 anchor 距离退化为 1。`failed` 计数只打印，不写盘。

---

## 14. Transfer 那条路径上 Face++ 与 Amazon 使用不同分母

`evaluation/regeneration/eval_arc2face_blackbox.py`：

- `:269-271` 原图无脸 → `continue`（Face++ 侧按 p6 规则正确排除，`levels` 不记录）
- 但 `:303-306` 的 Amazon 聚合无条件在后面执行：`amazon_passes == []` 时 `int(any([])) = 0` 被 append 进 `amazon_success_at_5_list` → 该样本在 Amazon 列记为**失败**
- `:182-187` `InvalidParameterException` → `return False`；`:193-197` 连续 10 次请求失败 → `return False` — API 故障记为身份不匹配
- 而 Face++ 侧的聚合在 `:311-319` 被 `continue` 跳过

两个效应都把 Table VI 的 Amazon 列（0.473 / 0.447）相对 Face++（0.946）往下压，而 p7 把这个差距解释为纯阈值差异。

---

## 15. Sec VII-C 的构造与代码不符

论文 p7：*"we draw a fresh random channel subset independently for **every training sample**, and we **average across the selected channels to form a single-channel image**, which **removes the need for the shuffling components** altogether."*

- FracFace 的 `random_per_sample` 路径（`attacks/channel_experiments/train_ours.py:67-69` → `methods/fracface_fixed/data2npy.py:64-69`）输出 **81 通道**，没有做平均；`methods/fracface_fixed/utils/fractal_utils.py:141-147` 里 fractal shuffle 仍然在应用（`fixed_channel=False` 只是换成不 seed 的 rng），即被随机化的正是论文说要移除的组件
- PartialFace 侧的随机是**每 batch 一次**：`attacks/partialface/processing_utils.py:105` `selection = torch.randperm(54).reshape(n_subset, 9)` 在 `create_channel_subsets` 内，每次调用一次；`attacks/channel_experiments/README.md:35` 自己写着 *"random mode draws one new 6x9 partition per batch"*，`train_ours.py:110-116` batch=256
- 这些 random 训练的 student 评测时又回到固定 secret：`dump_reconstructions.py:74` `fixed_channel = args.channel_mode == "fixed"`，artifact 目录名即 `*_random_train_fixed_test`

---

# 附：已核验一致的部分

- `data_splits/index.txt`：88,686 行 / 10,572 身份；8,457 train (70,692 图) / 2,115 val (17,994 图)，身份交集为 0 — 与 p4 "~10K identities and 90K images"、"80-20"、p5 "hold-out dataset size is 2115" 一致
- Eq(2) 在三个主 PPFR 攻击中逐字实现；teacher 全程 frozen（embedding 离线预抽，只从磁盘读）
- Eq(5) 忠实实现（L1，`pretrained=False` 确为未训练权重）；U-Net 与 ours 的判定条件与分母一致，两侧均单次尝试，上传前均 resize 到 256×256 LANCZOS JPEG
- 300 张评测图与训练集无污染（0/300 在 train，280 个评测身份 0 个出现在 train）
- Face++ "最严阈值" 规则在所有 scorer 中实现正确；排除规则极性正确（`image_file1` = 生成图、`image_file2` = 原图，故 `faces2 == []` → 排除、`faces1 == []` → Failed）
- Table VIII / IX 全部单元格可从 `results/*_converge.json` 精确复现；p8 声明的非对称性（gallery 恒用 frozen recognizer；query 侧 before-attack 与 upper-bound 用 frozen、after-attack 用 student）在 `g1_eval.py:67-72` 实现正确；Recall@1 / Recall@0.5% / Recall@0.5%_set / rank_best 定义与正文一致，K = `ceil(0.005 × 10723)` = 54
- Sec VII-B 迁移攻击的训练/推理分离正确：训练只用 MinusFace + 高通代理和 CASIA train 行，对外方 template 的推理是纯前向、无 optimizer
- PartialFace 的变换本身与官方 TFace 代码逐字一致（仅多 `fixed_channel` 开关）；MinusFace stage-1 checkpoint 确为训练过的模型（`num_batches_tracked = 117336`），`[5]` 索引取的是正确的 3 通道 protected 表示
- Table II 的 top-1 recall 公式与矩阵朝向（行 = query、列 = key）正确

---

# 附：已过滤的条目

**因"代码在时间上被改过、committed 状态不代表产出数字的那次运行"而过滤：**

- `attacks/fracface/insight_train.py` 的 committed 状态是 256 图 / 1 epoch / val 20 图 / 存 `student_mini.pth`
- `log/insight_student_embeddings_val_epoch{4}.pkl` 是 f-string 里的字面量 4，且被 `insight_train.py` 与 `insight_train_shuffle.py`（100 epoch + 50% block-drop）共用同一路径
- `attacks/partialface/insight_train.py:250` 的 val dump 被注释掉、`:34-40` 的 `Conv2d(27,3)` adapter 被注释掉、存的是 `student_flatten.pth` 而 `insight_test.py` 找 `student.pth`
- `eval_arc2face_fracface.py:108` / `_minus.py:109` 的 `compare_face_amazon` 被改成 `return -1`（Table V 的 MinusFace 0.98 / FracFace 0.92 无法从 committed 代码重跑）
- `eval_arc2face_blackbox.py:267` 是 `for i in range(1):` 而 `:262` 生成 5 张，使 committed 状态下 Table VI 两列（`at_least_one_passed`、`amazon_success_at_5`）退化为单张结果；三个主实验脚本对应位置都是 `range(5)`（`_fracface.py:248`、`_partial.py:234`、`_minus.py:238`）
- `attacks/canfg/extract_embeddings.py:25` 的 `root` 指向 `protected_A/`（notebook 版有 `.replace(...)` 换成原图）
- 无 ~800 图配置（Sec VII-A 只有 256 图那一档存在）
- `data_splits/index.txt` 由 `notebooks/fracface/split.ipynb` 从已有的 PartialFace student dump 反推（身份不重叠本身成立）

- `evaluation/results/res_*.pkl` 是 Table III 唯一留在盘上的原始记录，但只有 PartialFace 一行对得上（重算：0.9880 / 0.9802 / 0.9834 / 0.9856 vs 论文 0.988 / 0.980 / 0.983 / 0.986，四位小数全中，说明重算口径与论文一致）；同口径下 MinusFace 差 0.003–0.006（0.9920 / 0.9735 / 0.9836 / 0.9886 vs 0.987 / 0.974 / 0.981 / 0.983），FracFace 的 Pass@1e-5 差 5.8 点（0.8854 vs 0.943）。即盘上 artifact 不是产出论文数字的那次运行

**因量级过小或非结果性而过滤：**

- MinusFace / PartialFace 的攻击对已对齐的 CASIA 图再跑一次 `DeepFace.extract_faces(p, detector_backend="opencv", enforce_detection=False)`（`attacks/minusface/insight_train.py:124`、`attacks/partialface/insight_train.py:106`），FracFace 直接 `Image.open`（`attacks/fracface/insight_train.py:113`）—— 论文正文与 Table II 标题都没有声明跨方法预处理一致，故不构成 paper-vs-code 差异

- Success@5 分母漏掉"5 张全部检测失败"的样本（`eval_arc2face_fracface.py:280-285` 的 `continue` 在 append 之前）— 实测 998 vs 1000，约 2/1000
- Table IV/VI "optimal threshold" 是 `np.arange(0.1, 0.3, 0.01)` 20 个候选取 max、在同一 6000 对上拟合 — 报告值已在 0.99 附近，影响有界
- Table III MinusFace 行用 `guidance_scale=2.5`，FracFace/PartialFace 用 `3.0`
- Table VII 第一列（0.680 / 0.850 / 1.000）未实现 FracFace 的 Protection(%) 公式，属手工转录
- Table I（SSIM/PSNR/MSE/FS）无实现代码（motivating table）
- Fig 2b 的 StyleGAN 对比只有一张手工 `stylegan.png` 和单个身份
- de-id 侧"与 protection 训练集 disjoint"无法证实（四个变体均用第三方发布权重，其训练语料未记录）
