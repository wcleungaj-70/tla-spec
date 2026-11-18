# DMA评审跟踪

[返回接口目录](../README.md)

<a name="dma-004"></a>

## DMA-004 · tla.copy_l1_to_l0

[查看规格](../spec/dma.md#api-tla-copy-l1-to-l0)

- 确认物理 K 分区字段的单位和有效组合，补充按 L0A/L0B 分开的负例。

- `src`、`dst`：确认目标模板的具体元素和布局组合后纳入泛化测试。

- `params.k_part_idx`、`params.k_part_ceil`、`params.k_part_loop`、`params.k_part_actual`：按目标内核的分区协议补充运行范围和字段间关系。

结论：本轮范围已核对，按规格限定组合转测；其他能力本轮不开放。

<a name="dma-005"></a>

## DMA-005 · tla.make_partition_view

[查看规格](../spec/dma.md#api-tla-make-partition-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-006"></a>

## DMA-006 · tla.make_strided_view

[查看规格](../spec/dma.md#api-tla-make-strided-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-007"></a>

## DMA-007 · tla.make_gather_scatter_view

[查看规格](../spec/dma.md#api-tla-make-gather-scatter-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-008"></a>

## DMA-008 · tla.load_view

[查看规格](../spec/dma.md#api-tla-load-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-009"></a>

## DMA-009 · tla.store_view

[查看规格](../spec/dma.md#api-tla-store-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-010"></a>

## DMA-010 · tla.set_pad_value

[查看规格](../spec/dma.md#api-tla-set-pad-value)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-011"></a>

## DMA-011 · tla.set_post_copy

[查看规格](../spec/dma.md#api-tla-set-post-copy)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-012"></a>

## DMA-012 · tla.stride_store

[查看规格](../spec/dma.md#api-tla-stride-store)

原评审问题：是否纳入本轮规格并补齐实现。

结论：已关闭。确认不需要此接口，退出规格范围。

<a name="dma-013"></a>

## DMA-013 · tla.stride_load

[查看规格](../spec/dma.md#api-tla-stride-load)

原评审问题：是否纳入本轮规格并补齐实现。

结论：已关闭。确认不需要此接口，退出规格范围。

<a name="ricky-scope"></a>

## Ricky 接口：950 转测范围依据

本次为源码审查，未执行设备精度测试。TA 提交 `008dff39bdcc908d6706946189b0c53c5257e4d4`，NPU IR 提交 `cfe3feea3da5cbc603cb71221958cbf6e7bb9f95`。工作区有既存改动，相关 DMA 前端和后端模板与提交核对；本次不修改实现。进度仅维护 README，裁剪范围见规格页。

- `tla.load_gm_to_l1`：BF16 2D ND→4D NZ，DSA 无 params 调用。交付 params 仅 enable_nd2nz=True（默认且唯一支持值），False 必须拒绝。当前前端固定生成 nd2nz，尚无此字段，字段适配待实现。TODO：l2_cache_ctrl、enable_dn2nz 后续支持；缓存控制已有后端实现，本轮排除只是范围裁剪。旧 enable_nz2nd/enable_nz2dn 不是 enable_dn2nz；窗口行参数本轮也不开放。
- `tla.copy_l1_to_l0`：同类型 FP16/BF16、4D 物理 tile，L0A/L0B 分别验证，保留 is_transpose 及六个必填分块字段。先用整 K 或等长无尾块：k_part=k_part_ceil=k_part_actual 为 16 的正整数倍，K 整除 k_part，k_part_loop=K/k_part，0<=k_part_idx<k_part_loop，m 为对应 M/N 的 16 倍数且不越界。满足模板的 512B 地址对齐、fractal 对齐 offset 和容量限制。删除 actual_batch；TODO：尾块、其他类型和 batch 另行验证。同步及 L0 生命周期由调用方保证。

源码与用例证据（测试定义不代表本轮已通过）：

- [kernels/dsa/dsa_kernel.py](https://github.com/triton-lang/triton-ascend/blob/008dff39bdcc908d6706946189b0c53c5257e4d4/third_party/ascend/unittest/tla_ut/kernels/dsa/dsa_kernel.py)
- [dma/nd2nz.py](https://github.com/triton-lang/triton-ascend/blob/008dff39bdcc908d6706946189b0c53c5257e4d4/third_party/ascend/unittest/tla_ut/dma/nd2nz.py)
- [cube/mmad_l0.py](https://github.com/triton-lang/triton-ascend/blob/008dff39bdcc908d6706946189b0c53c5257e4d4/third_party/ascend/unittest/tla_ut/cube/mmad_l0.py)
- [cube/mmad_l0_reuse.py](https://github.com/triton-lang/triton-ascend/blob/008dff39bdcc908d6706946189b0c53c5257e4d4/third_party/ascend/unittest/tla_ut/cube/mmad_l0_reuse.py)
- [NPU IR bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp](https://gitcode.com/Ascend/AscendNPU-IR/blob/cfe3feea3da5cbc603cb71221958cbf6e7bb9f95/bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp)
- [NPU IR bishengir/lib/Template/lib/RegBase/Cube/l12l0.cpp](https://gitcode.com/Ascend/AscendNPU-IR/blob/cfe3feea3da5cbc603cb71221958cbf6e7bb9f95/bishengir/lib/Template/lib/RegBase/Cube/l12l0.cpp)
