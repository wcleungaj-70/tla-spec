# Cube评审跟踪

[返回接口目录](../README.md)

<a name="cube-001"></a>

## CUBE-001 · tla.copy_ub_to_l1

[查看规格](../spec/cube.md#api-tla-copy-ub-to-l1)

- 确认 UB→L1 是否应在前端拒绝非 None 的填充配置，并明确 Vector 写入后由 Cube 消费时的同步约定。

结论：本轮范围已核对，按规格限定组合转测；其他能力本轮不开放。

<a name="cube-003"></a>

## CUBE-003 · tla.copy_l1_to_bt

[查看规格](../spec/cube.md#api-tla-copy-l1-to-bt)

- `src`、`dst`：确认目标模板的具体元素和布局组合后纳入泛化测试。

结论：本轮范围已核对，按规格限定组合转测；其他能力本轮不开放。

<a name="cube-004"></a>

## CUBE-004 · tla.copy_l1_to_fb

[查看规格](../spec/cube.md#api-tla-copy-l1-to-fb)

- 目前只有预留入口，建议在完成后端映射前标记为不可转测。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="cube-005"></a>

## CUBE-005 · tla.copy_l0c_to_ub

[查看规格](../spec/cube.md#api-tla-copy-l0c-to-ub)

- `params.pre_relu`：为额外激活模式补充目标支持和参数约束。

结论：本轮范围已核对，按规格限定组合转测；其他能力本轮不开放。

<a name="cube-006"></a>

## CUBE-006 · tla.copy_l0c_to_l1

[查看规格](../spec/cube.md#api-tla-copy-l0c-to-l1)

- `params.pre_relu`：为额外激活模式补充目标支持和参数约束。

结论：本轮范围已核对，按规格限定组合转测；其他能力本轮不开放。

<a name="cube-007"></a>

## CUBE-007 · tla.store_l0c_to_gm

[查看规格](../spec/cube.md#api-tla-store-l0c-to-gm)

- `params.pre_relu`：为额外激活模式补充目标支持和参数约束。

结论：本轮范围已核对，按规格限定组合转测；其他能力本轮不开放。

<a name="ricky-scope"></a>

## Ricky 接口：950 转测范围依据

本次为源码审查，未执行设备精度测试。TA 提交 `008dff39bdcc908d6706946189b0c53c5257e4d4`，NPU IR 提交 `cfe3feea3da5cbc603cb71221958cbf6e7bb9f95`。工作区有既存改动，相关 DMA 前端和后端模板与提交核对；本次不修改实现。进度仅维护 README，裁剪范围见规格页。

- `tla.copy_ub_to_l1`：BF16 UB→L1；保持已准备的物理布局，保留 DSA 使用的带步长 UB 子视图及带偏移 L1 子视图。交付仅 src/dst，TODO：pad_mode、pad_value、collapse_reassociation 后续支持。当前前端仍接收 params，签名裁剪待实现。跨核交接须显式同步。
- `tla.copy_l0c_to_ub`：仅 FP32→FP32，dma_mode=NZ2ND、dual_dst_mode=ROW_SPLIT、sub_block_idx=None。其他字段固定默认：NO_QUANT、NO_RELU、quant_scale=None、unit_flag_ctrl/group_id=None；属性缺省不等于 DISABLED。TODO：其他布局、量化、激活、缩放、切分和手动 UnitFlag 另行验证。
- `tla.copy_l1_to_ub`：同类型 FP16/BF16，4D NZ→2D ND，默认空配置。保留 dma/copy.py、dma/nd2nz.py、dma/nd2nz_window.py 的基础路径；其他类型和批量 5D 路径暂不纳入。AIV 读取前须同步并匹配接收子块。
- `tla.copy_l1_to_bt`：A5 已有 lowering 和模板，限定 FP16/BF16/FP32→FP32 或 INT32→INT32；n 是常量 1..1024，None 仅在源末维为合法常量时可用。rank 1..4，前导维为 1，末维 stride=1，静态正 strides/offset；容量至少 roundUp(n,16) 个元素，L1 32B/BT 64B 地址对齐且不越界。内部测试使用合法 BT 地址，不承诺 tla.alloc 支持 BT。动态 n、任意二维矩阵、FP16→FP16 BT 不纳入。旧 copy.py 只有 IR 构造证据。
- `tla.copy_l0c_to_l1`：仅 FP32→FP16、NZ2NZ、F322F16；固定 NO_RELU、quant_scale=None、channel_split=False、c0_pad_en=True、unit_flag_ctrl/group_id=None。对应 dma/fixpipe.py 的 l1_nz。TODO：其他类型、布局、channel_split、关闭 C0 padding、激活和手动 UnitFlag 不纳入本轮。
- `tla.store_l0c_to_gm`：NZ2ND，FP32→FP32/FP16/BF16 分别使用 NO_QUANT/F322F16/F322BF16；固定 NO_RELU、quant_scale=None、unit_flag_ctrl/group_id=None。对应 dma/fixpipe.py 的 gm_fp32/gm_fp16/gm_bf16。TODO：其他布局、INT32、缩放、激活和手动 UnitFlag 不纳入本轮。

源码与用例证据（测试定义不代表本轮已通过）：

- [kernels/dsa/dsa_kernel.py](https://github.com/triton-lang/triton-ascend/blob/008dff39bdcc908d6706946189b0c53c5257e4d4/third_party/ascend/unittest/tla_ut/kernels/dsa/dsa_kernel.py)
- [dma/copy.py](https://github.com/triton-lang/triton-ascend/blob/008dff39bdcc908d6706946189b0c53c5257e4d4/third_party/ascend/unittest/tla_ut/dma/copy.py)
- [dma/fixpipe.py](https://github.com/triton-lang/triton-ascend/blob/008dff39bdcc908d6706946189b0c53c5257e4d4/third_party/ascend/unittest/tla_ut/dma/fixpipe.py)
- [NPU IR bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp](https://gitcode.com/Ascend/AscendNPU-IR/blob/cfe3feea3da5cbc603cb71221958cbf6e7bb9f95/bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp)
- [NPU IR bishengir/lib/Template/lib/RegBase/Cube/l12bt.cpp](https://gitcode.com/Ascend/AscendNPU-IR/blob/cfe3feea3da5cbc603cb71221958cbf6e7bb9f95/bishengir/lib/Template/lib/RegBase/Cube/l12bt.cpp)
