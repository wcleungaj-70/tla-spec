# Cube评审跟踪

[返回接口目录](../README.md)

<a name="cube-001"></a>

## CUBE-001 · tla.copy_ub_to_l1

[查看规格](../spec/cube.md#api-tla-copy-ub-to-l1)

结论：本轮交付移除 `params`，仅保留 `src`、`dst`；填充和维度合并不纳入本次测试。保留 Vector→Cube 的同步要求。

TODO（后续支持，不阻塞本轮）：`pad_mode`、`pad_value`、`collapse_reassociation`。

支持度核对：

- [TLA dma.py](https://github.com/triton-lang/triton-ascend/blob/008dff39bdcc908d6706946189b0c53c5257e4d4/third_party/ascend/language/tla/dma.py#L658) 当前仍接收配置并传给 CopyOp，交付实现需同步移除该入口参数。
- NPU IR `cfe3feea3da5cbc603cb71221958cbf6e7bb9f95` 的 `bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp` 中，UB→L1 的 padding 传参开关为 `{false, false}`；不能把 IR 能携带 padding 当作目标搬运已支持。
- `collapse_reassociation` 在 `HIVMDMAOps.cpp`、`LiftLowestStride.cpp` 有读取和维护逻辑；TLA `tla_ut/dma/copy.py` 有 attribute 检查，不等同于完整 UB→L1 功能测试。
- 本次为源码和测试定义核对，未执行设备验证；未修改编译器或 TLA 实现仓库。

<a name="cube-003"></a>

## CUBE-003 · tla.copy_l1_to_bt

[查看规格](../spec/cube.md#api-tla-copy-l1-to-bt)

- `src`、`dst`：确认目标模板的具体元素和布局组合后纳入泛化测试。

结论：待决。

<a name="cube-004"></a>

## CUBE-004 · tla.copy_l1_to_fb

[查看规格](../spec/cube.md#api-tla-copy-l1-to-fb)

- 目前只有预留入口，建议在完成后端映射前标记为不可转测。

结论：待决。

<a name="cube-005"></a>

## CUBE-005 · tla.copy_l0c_to_ub

[查看规格](../spec/cube.md#api-tla-copy-l0c-to-ub)

- `params.pre_relu`：为额外激活模式补充目标支持和参数约束。

结论：待决。

<a name="cube-006"></a>

## CUBE-006 · tla.copy_l0c_to_l1

[查看规格](../spec/cube.md#api-tla-copy-l0c-to-l1)

- `params.pre_relu`：为额外激活模式补充目标支持和参数约束。

结论：待决。

<a name="cube-007"></a>

## CUBE-007 · tla.store_l0c_to_gm

[查看规格](../spec/cube.md#api-tla-store-l0c-to-gm)

- `params.pre_relu`：为额外激活模式补充目标支持和参数约束。

结论：待决。
