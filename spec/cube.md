# Cube接口规格

[返回接口目录](../README.md)

<a name="api-tla-copy-ub-to-l1"></a>

## tla.copy_ub_to_l1

本轮交付签名：`tla.copy_ub_to_l1(src, dst)`

返回：None。

编译表示：hivm.hir.copy。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32 | 见共同约束。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：L1<br>数据类型：与 src 相同 | 见共同约束。 | 必填 |

共同约束：此接口在前端固定为 UB→L1，降低为 hivm.hir.copy。RegBase UB→L1 模板注册的元素类型比通用 CopyOp 窄。本轮不提供配置参数，要求源和目的形状相容。目标末维连续，调用前应准备相容的物理布局。Vector 生产的数据跨核供 Cube 消费时需要显式同步。

**本轮范围：** 不提供 `params`；填充和维度合并配置不纳入本次测试。现有源码仍有可选配置对象，交付实现需同步收敛签名。

**TODO：** 后续支持 `pad_mode`、`pad_value`、`collapse_reassociation`，补齐目标约束及端到端测试；当前支持度和实现差异见 [CUBE-001](../tracking/cube.md#cube-001)。

**调用示例**

x 指向预先按目标物理布局准备的 FP16 数据。若另一核随后消费 L1，须在交接处同步。

```python
@tla.jit
def load_tile_to_l1(x):
    dst_l1 = tla.alloc((1, 1, 16, 16), tl.float16, space="l1")
    with tla.scope.vector():
        src_ub = tla.alloc((1, 1, 16, 16), tl.float16)
        tla.load_gm_to_ub(tla.view(x, (1, 1, 16, 16)), src_ub)
        tla.copy_ub_to_l1(src_ub, dst_l1)
```

仓库测试：[tla_ut/dma/copy.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/copy.py#L19)。

<a name="api-tla-copy-l1-to-ub"></a>

## tla.copy_l1_to_ub

签名：`tla.copy_l1_to_ub(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.l12ub。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L1<br>数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 同 dtype。RegBase 模板形状为 4D NZ→2D ND 或 5D NZ→3D ND。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 同 dtype。RegBase 模板形状为 4D NZ→2D ND 或 5D NZ→3D ND。 | 必填 |
| `params` | 控制搬运的配置 | tla.CopyL12UBParams 或 None | None / tla.CopyL12UBParams 实例 | None 创建默认配置。 | None |

**调用示例**

源为已就绪的 NZ tile，目标为同类型的 ND 缓冲区。

```python
@tla.jit
def example(src_l1, dst_ub):
    with tla.scope.cube():
        tla.copy_l1_to_ub(src_l1, dst_ub)
```

仓库测试：[tla_ut/dma/copy.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/copy.py#L30)。

<a name="api-tla-copy-l1-to-bt"></a>

## tla.copy_l1_to_bt

签名：`tla.copy_l1_to_bt(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.l12bt。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L1<br>数据类型：待确认目标支持集合 | L12BTOp 校验地址空间，ODS 未枚举元素类型。偏置格式与目标实现有关。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：BT<br>数据类型：待确认目标支持集合 | L12BTOp 校验地址空间，ODS 未枚举元素类型。偏置格式与目标实现有关。 | 必填 |
| `params` | 控制搬运的配置 | tla.CopyL12BTParams 或 None | None / tla.CopyL12BTParams 实例 | None 创建默认配置。 | None |
| `params.n` | 传给 L12BTOp 的 N 轴长度 | int / 标量整数 tl.tensor / None | None 或正整数 | None 使用 src.shape[-1]。实际读取不能越界。 | None |

共同约束：L12BTOp 的 verifier 约束源为 L1、目的为 BiasBUF。TableGen 未枚举元素类型，小测试中的 FP16/FP32 只能作为构造 IR 的用例。

仓库 IR 构造用例：[tla_ut/dma/copy.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/copy.py#L45)。

<a name="api-tla-copy-l1-to-fb"></a>

## tla.copy_l1_to_fb

签名：`tla.copy_l1_to_fb(src, dst, params=None)`

**预留入口。** 当前调用会抛出 NotImplementedError。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L1<br>数据类型：无可执行签名 | 前端校验空间和配置对象后总是抛 NotImplementedError，不生成 DMA op。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：FB<br>数据类型：无可执行签名 | 前端校验空间和配置对象后总是抛 NotImplementedError，不生成 DMA op。 | 必填 |
| `params` | 控制搬运的配置 | tla.CopyL12FBParams 或 None | None / tla.CopyL12FBParams 实例 | None 创建默认配置。 | None |

<a name="api-tla-copy-l0c-to-ub"></a>

## tla.copy_l0c_to_ub

签名：`tla.copy_l0c_to_ub(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.fixpipe。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L0C<br>数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / float32<br>B64：不支持 | Ascend950/960 的 UB 目标。双目标时 sub_block_idx 必须为 None。 源/目的组合见 params.pre_quant。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：<br>B8：int8<br>B16：float16 / bfloat16<br>B32：int32 / float32<br>B64：不支持 | Ascend950/960 的 UB 目标。双目标时 sub_block_idx 必须为 None。 源/目的组合见 params.pre_quant。 | 必填 |
| `params` | 控制搬运的配置 | tla.FixpipeL0C2UBParams 或 None | None / tla.FixpipeL0C2UBParams 实例 | None 创建默认配置。 | None |
| `params.dma_mode` | 选择输出布局 | 编译期 tla.FixpipeDMAMode | NZ2NZ / NZ2ND / NZ2DN | NZ2DN 需要 dav-c310 或 dav-920r1。 | FixpipeDMAMode.NZ2NZ |
| `params.dual_dst_mode` | 选择单目标或双 AIV 子块的切分方式 | 编译期 tla.FixpipeDualDstMode | NO_DUAL / ROW_SPLIT / COLUMN_SPLIT | 双目标仅在支持的 L0C→UB 路径上使用。 | FixpipeDualDstMode.NO_DUAL |
| `params.sub_block_idx` | 选择单目标 AIV 子块 | 编译期 tla.FixpipeSubBlock 或 None | None / SUB_BLOCK_0 / SUB_BLOCK_1 | ROW_SPLIT 或 COLUMN_SPLIT 时必须为 None。 | FixpipeSubBlock.SUB_BLOCK_0 |
| `params.pre_quant` | 选择写出时的类型转换 | 编译期 tla.FixpipePreQuantMode | NO_QUANT：float32→float32 / int32→int32<br>F322F16：float32→float16<br>F322BF16：float32→bfloat16<br>S322I8：int32→int8<br>QF322F32_PRE：float32→float32 | src/dst 必须匹配所选转换。 | FixpipePreQuantMode.NO_QUANT |
| `params.pre_relu` | 选择写出前的激活 | 编译期 tla.FixpipePreReluMode | NO_RELU / NORMAL_RELU（通用选项） | 前端还暴露 LEAKY_RELU / P_RELU。具体目标和参数组合需单独核验。 | FixpipePreReluMode.NO_RELU |
| `params.quant_scale` | 提供量化或缩放系数 | int / float / 标量 tl.tensor / None | None 或可转为 float32 的标量 | 具体模式决定是否消费该操作数。 | None |
| `params.unit_flag_ctrl` | 选择累加器 UnitFlag 的控制方式 | 编译期 tla.UnitFlagMode / int / None | None / 0（DISABLED）/ 2（ENABLED_WITHOUT_UPDATE）/ 3（ENABLED_WITH_UPDATE） | 1 保留。None 省略属性。手动模式可用 unit_flag=False 关闭自动推断。 | None |
| `params.unit_flag_group_id` | 指定配对操作的 UnitFlag 组号 | 编译期 int 或 None | None 或 −2⁶³ 至 2⁶³−1（IR 编码） | 与生产者及消费者协议一致。编码域不代表芯片可用 ID 集。 | None |

**调用示例**

示例使用默认 NZ→NZ 和 NO_QUANT。

```python
@tla.jit
def example(acc_l0c, dst_ub):
    with tla.scope.cube():
        tla.copy_l0c_to_ub(acc_l0c, dst_ub, tla.FixpipeL0C2UBParams())
```

仓库测试：[tla_ut/cube/gemm_bias.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/cube/gemm_bias.py#L35)。

<a name="api-tla-copy-l0c-to-l1"></a>

## tla.copy_l0c_to_l1

签名：`tla.copy_l0c_to_l1(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.fixpipe。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L0C<br>数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / float32<br>B64：不支持 | channel_split 为 True 时需要 Ascend950/960。 源/目的组合见 params.pre_quant。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：L1<br>数据类型：<br>B8：int8<br>B16：float16 / bfloat16<br>B32：int32 / float32<br>B64：不支持 | channel_split 为 True 时需要 Ascend950/960。 源/目的组合见 params.pre_quant。 | 必填 |
| `params` | 控制搬运的配置 | tla.FixpipeL0C2L1Params 或 None | None / tla.FixpipeL0C2L1Params 实例 | None 创建默认配置。 | None |
| `params.dma_mode` | 选择输出布局 | 编译期 tla.FixpipeDMAMode | NZ2NZ / NZ2ND / NZ2DN | NZ2DN 需要 dav-c310 或 dav-920r1。 | FixpipeDMAMode.NZ2NZ |
| `params.pre_quant` | 选择写出时的类型转换 | 编译期 tla.FixpipePreQuantMode | NO_QUANT：float32→float32 / int32→int32<br>F322F16：float32→float16<br>F322BF16：float32→bfloat16<br>S322I8：int32→int8<br>QF322F32_PRE：float32→float32 | src/dst 必须匹配所选转换。 | FixpipePreQuantMode.NO_QUANT |
| `params.pre_relu` | 选择写出前的激活 | 编译期 tla.FixpipePreReluMode | NO_RELU / NORMAL_RELU（通用选项） | 前端还暴露 LEAKY_RELU / P_RELU。具体目标和参数组合需单独核验。 | FixpipePreReluMode.NO_RELU |
| `params.channel_split` | 启用目标 L1 的通道切分 | 编译期 bool | False / True | True 需要 Ascend950/960。 | False |
| `params.c0_pad_en` | 允许 L1 目的布局的 C0 填充 | 编译期 bool | False / True | 根据目的 tile 和布局使用。 | True |
| `params.unit_flag_ctrl` | 选择累加器 UnitFlag 的控制方式 | 编译期 tla.UnitFlagMode / int / None | None / 0（DISABLED）/ 2（ENABLED_WITHOUT_UPDATE）/ 3（ENABLED_WITH_UPDATE） | 1 保留。None 省略属性。手动模式可用 unit_flag=False 关闭自动推断。 | None |
| `params.unit_flag_group_id` | 指定配对操作的 UnitFlag 组号 | 编译期 int 或 None | None 或 −2⁶³ 至 2⁶³−1（IR 编码） | 与生产者及消费者协议一致。编码域不代表芯片可用 ID 集。 | None |
| `params.quant_scale` | 提供量化或缩放系数 | int / float / 标量 tl.tensor / None | None 或可转为 float32 的标量 | 具体模式决定是否消费该操作数。 | None |

**调用示例**

目的 L1 使用与默认 NZ→NZ 兼容的布局。

```python
@tla.jit
def example(acc_l0c, dst_l1):
    with tla.scope.cube():
        tla.copy_l0c_to_l1(acc_l0c, dst_l1)
```

仓库测试：[tla_ut/dma/fixpipe.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/fixpipe.py#L69)。

<a name="api-tla-store-l0c-to-gm"></a>

## tla.store_l0c_to_gm

签名：`tla.store_l0c_to_gm(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.fixpipe。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L0C<br>数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / float32<br>B64：不支持 | 由 dma_mode 决定输出布局。 源/目的组合见 params.pre_quant。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：GM<br>数据类型：<br>B8：int8<br>B16：float16 / bfloat16<br>B32：int32 / float32<br>B64：不支持 | 由 dma_mode 决定输出布局。 源/目的组合见 params.pre_quant。 | 必填 |
| `params` | 控制搬运的配置 | tla.FixpipeL0C2GMParams 或 None | None / tla.FixpipeL0C2GMParams 实例 | None 创建默认配置。 | None |
| `params.dma_mode` | 选择输出布局 | 编译期 tla.FixpipeDMAMode | NZ2NZ / NZ2ND / NZ2DN | NZ2DN 需要 dav-c310 或 dav-920r1。 | FixpipeDMAMode.NZ2NZ |
| `params.pre_quant` | 选择写出时的类型转换 | 编译期 tla.FixpipePreQuantMode | NO_QUANT：float32→float32 / int32→int32<br>F322F16：float32→float16<br>F322BF16：float32→bfloat16<br>S322I8：int32→int8<br>QF322F32_PRE：float32→float32 | src/dst 必须匹配所选转换。 | FixpipePreQuantMode.NO_QUANT |
| `params.pre_relu` | 选择写出前的激活 | 编译期 tla.FixpipePreReluMode | NO_RELU / NORMAL_RELU（通用选项） | 前端还暴露 LEAKY_RELU / P_RELU。具体目标和参数组合需单独核验。 | FixpipePreReluMode.NO_RELU |
| `params.quant_scale` | 提供量化或缩放系数 | int / float / 标量 tl.tensor / None | None 或可转为 float32 的标量 | 具体模式决定是否消费该操作数。 | None |
| `params.unit_flag_ctrl` | 选择累加器 UnitFlag 的控制方式 | 编译期 tla.UnitFlagMode / int / None | None / 0（DISABLED）/ 2（ENABLED_WITHOUT_UPDATE）/ 3（ENABLED_WITH_UPDATE） | 1 保留。None 省略属性。手动模式可用 unit_flag=False 关闭自动推断。 | None |
| `params.unit_flag_group_id` | 指定配对操作的 UnitFlag 组号 | 编译期 int 或 None | None 或 −2⁶³ 至 2⁶³−1（IR 编码） | 与生产者及消费者协议一致。编码域不代表芯片可用 ID 集。 | None |

**调用示例**

dst_gm 是已构造的 GM 缓冲区，形状与默认 NZ→NZ 模式相容。

```python
@tla.jit
def example(acc_l0c, dst_gm):
    with tla.scope.cube():
        tla.store_l0c_to_gm(acc_l0c, dst_gm)
```

仓库测试：[tla_ut/cube/allocation.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/cube/allocation.py#L47)。

<a name="api-tla-cube-mmad-mx"></a>

## tla.cube.mmad_mx

签名：`tla.cube.mmad_mx(a, b, scale_a, scale_b, acc, *, format_a=None, format_b=None, transpose_a=False, transpose_b=False, clear_accum=False, bias=None, real_m=None, real_k=None, real_n=None)`

返回：None。

编译表示：hivm.hir.mmadmxL1。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `a` | 第一个操作数 | tla.buffer | 地址空间：L1<br>数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：不支持<br>B32：不支持<br>B64：不支持 | 元素按 FP8 格式或 packed E2M1 解读。双边须同为 FP4 或同为 FP8。 采用逻辑 2D 或物理 4D NZ。 | 必填 |
| `b` | 第二个操作数 | tla.buffer | 地址空间：L1<br>数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：不支持<br>B32：不支持<br>B64：不支持 | 元素按 FP8 格式或 packed E2M1 解读。双边须同为 FP4 或同为 FP8。 采用逻辑 2D 或物理 4D NZ。 | 必填 |
| `scale_a` | A 的 E8M0 块缩放表 | tla.buffer | 地址空间：L1<br>数据类型：int8 / uint8（E8M0 编码） | 每 32 个逻辑 K 取一个因子，逻辑形状 (M, K/32)。4D zZ/nN 的奇数组数对齐到偶数。 | 必填 |
| `scale_b` | B 的 E8M0 块缩放表 | tla.buffer | 地址空间：L1<br>数据类型：int8 / uint8（E8M0 编码） | 每 32 个逻辑 K 取一个因子，逻辑形状 (N, K/32)。4D zZ/nN 的奇数组数对齐到偶数。 | 必填 |
| `acc` | 累加器 | tla.buffer | 地址空间：L0C<br>数据类型：<br>B8：不支持<br>B16：不支持<br>B32：float32<br>B64：不支持 | 逻辑形状须等于 (real_m, real_n)。clear_accum=False 时读取已有累加值。 | 必填 |
| `format_a` | A 的 MX 编码格式 | tla.MxFormat 或 None | None / E5M2 / E4M3 / E2M1 | None 仅能从带类型的 FP8 推断。E2M1 每字节装两个值。 | None |
| `format_b` | B 的 MX 编码格式 | tla.MxFormat 或 None | None / E5M2 / E4M3 / E2M1 | None 仅能从带类型的 FP8 推断。E2M1 每字节装两个值。 | None |
| `transpose_a` | 是否在乘法的逻辑索引中转置 A | 编译期 bool | False / True | 转置后须满足逻辑形状。 | False |
| `transpose_b` | 是否在乘法的逻辑索引中转置 B | 编译期 bool | False / True | 转置后须满足逻辑形状。 | False |
| `clear_accum` | 是否从零开始计算 | 编译期 bool | False / True | 为 True 时从零开始，False 累加到已有值。 | False |
| `bias` | 可选的逐通道偏置 | tla.buffer 或 None | None 或 L1 buffer<br>数据类型：同 acc | 形状 (N,)、(1,N) 或等价 4D zN。 | None |
| `real_m` | 实际计算的 M 维长度 | 编译期 int 或 None | None 或 16 的正整数倍，不超过对应 tile 维度 | None 使用全长。 | None |
| `real_k` | 实际使用的 K 维长度 | 编译期 int 或 None | None 或 16 的正整数倍，不超过对应 tile 维度 | None 使用全长。MX 的 packed FP4 real_k 用字节计，逻辑 K 按每字节两个值计算。 | None |
| `real_n` | 实际计算的 N 维长度 | 编译期 int 或 None | None 或 16 的正整数倍，不超过对应 tile 维度 | None 使用全长。 | None |

**调用示例**

Ascend950PR。输入为布局相容的 FP8 L1 tile，缩放表为 E8M0，累加缓冲区为 float32 L0C。

```python
@tla.jit
def example(a_l1, b_l1, scale_a, scale_b, acc_l0c):
    with tla.scope.cube():
        tla.cube.mmad_mx(a_l1, b_l1, scale_a, scale_b, acc_l0c, clear_accum=True)
```

<a name="api-tla-scope-cube"></a>

## tla.scope.cube

签名：`tla.scope.cube()`

返回：无独立返回值。通过 with 声明内核区域。

将区域标记为 Cube 核模式。区域之间的跨核数据交接需要显式同步。

**调用示例**

```python
@tla.jit
def example(a_l1, b_l1, acc_l0c):
    with tla.scope.cube():
        tla.cube.mmad(a_l1, b_l1, acc_l0c, clear_accum=True)
```

仓库测试：[tla_ut/dma/ctrl.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/ctrl.py#L19)。

<a name="api-tla-cube-mmad"></a>

## tla.cube.mmad

签名：`tla.cube.mmad(a, b, acc, transpose_a=False, transpose_b=False, clear_accum=False, unit_flag_ctrl=None, *, unit_flag_group_id=None, real_m=None, real_k=None, real_n=None)`

返回：None。

编译表示：hivm.hir.mmadL1。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `a` | 第一个操作数 | tla.buffer | 地址空间：L1<br>数据类型：已确认 float16 / bfloat16 / float32 / int8 的常规签名，详见其他约束 | a/b 元素类型相同，转置后的 K 相等。浮点输入累加到 float32，常规 int8 输入累加到 int32。 采用逻辑 2D 或物理 4D NZ。 | 必填 |
| `b` | 第二个操作数 | tla.buffer | 地址空间：L1<br>数据类型：已确认 float16 / bfloat16 / float32 / int8 的常规签名，详见其他约束 | a/b 元素类型相同，转置后的 K 相等。浮点输入累加到 float32，常规 int8 输入累加到 int32。 采用逻辑 2D 或物理 4D NZ。 | 必填 |
| `acc` | 累加器 | tla.buffer | 地址空间：L0C<br>数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / float32<br>B64：不支持 | 逻辑形状须等于 (real_m, real_n)。clear_accum=False 时读取已有累加值。 | 必填 |
| `transpose_a` | 是否在乘法的逻辑索引中转置 A | 编译期 bool | False / True | 转置后须满足逻辑形状。 | False |
| `transpose_b` | 是否在乘法的逻辑索引中转置 B | 编译期 bool | False / True | 转置后须满足逻辑形状。 | False |
| `clear_accum` | 是否从零开始计算 | 编译期 bool | False / True | 为 True 时从零开始，False 累加到已有值。 | False |
| `unit_flag_ctrl` | 选择累加器 UnitFlag 的控制方式 | 编译期 tla.UnitFlagMode / int / None | None / 0（DISABLED）/ 2（ENABLED_WITHOUT_UPDATE）/ 3（ENABLED_WITH_UPDATE） | 1 保留。None 省略属性。手动模式可用 unit_flag=False 关闭自动推断。 | None |
| `unit_flag_group_id` | 指定配对操作的 UnitFlag 组号 | 编译期 int 或 None | None 或 −2⁶³ 至 2⁶³−1（IR 编码） | 与生产者及消费者协议一致。编码域不代表芯片可用 ID 集。 | None |
| `real_m` | 实际计算的 M 维长度 | 编译期 int 或 None | None 或 16 的正整数倍，不超过对应 tile 维度 | None 使用全长。 | None |
| `real_k` | 实际使用的 K 维长度 | 编译期 int 或 None | None 或 16 的正整数倍，不超过对应 tile 维度 | None 使用全长。 | None |
| `real_n` | 实际计算的 N 维长度 | 编译期 int 或 None | None 或 16 的正整数倍，不超过对应 tile 维度 | None 使用全长。 | None |

**调用示例**

A、B 为布局相容的 L1 tile，例如 float16 输入，acc 为 float32 L0C。

```python
@tla.jit
def example(a_l1, b_l1, acc_l0c):
    with tla.scope.cube():
        tla.cube.mmad(a_l1, b_l1, acc_l0c, clear_accum=True)
```

仓库测试：[tla_ut/cube/gemm_bias.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/cube/gemm_bias.py#L34)。

<a name="api-tla-set-ctrl"></a>

## tla.set_ctrl

签名：`tla.set_ctrl(enable, idx)`

返回：None。

编译表示：hivm.hir.set_ctrl。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `enable` | 是否置位 | 编译期 bool | False / True | — | 必填 |
| `idx` | CTRL bit 索引 | 编译期 int | 0 至 63（64 位 CTRL 的 bit 索引） | 仅使用目标定义的位。不会自动恢复状态。 | 必填 |

**调用示例**

示例为当前表中记录的 RegBase HF32 位。

```python
@tla.jit
def example():
    with tla.scope.cube():
        tla.set_ctrl(True, 46)
```

仓库测试：[tla_ut/dma/ctrl.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/ctrl.py#L20)。
