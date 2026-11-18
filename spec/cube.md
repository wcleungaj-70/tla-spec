# Cube接口规格

[返回接口目录](../README.md)

本轮标有“950 交付范围”的接口按该段限定转测；表内其他历史类型范围不属于本轮承诺。裁剪参数的默认行为仍须在实现内部保持，不能连同其语义删除。

<a name="api-tla-copy-ub-to-l1"></a>

## tla.copy_ub_to_l1

**本轮 950 交付范围（TA转测）：** BF16 UB→L1；保持已准备的物理布局，保留 DSA 使用的带步长 UB 子视图及带偏移 L1 子视图。交付仅 src/dst，TODO：pad_mode、pad_value、collapse_reassociation 后续支持。当前前端仍接收 params，签名裁剪待实现。跨核交接须显式同步。

交付签名（待前端适配）：`tla.copy_ub_to_l1(src, dst)`

返回：None。

编译表示：hivm.hir.copy。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32 | 见共同约束。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：L1<br>数据类型：与 src 相同 | 见共同约束。 | 必填 |

共同约束：此接口在前端固定为 UB→L1，降低为 hivm.hir.copy。RegBase UB→L1 模板注册的元素类型比通用 CopyOp 窄。默认配置要求源和目的形状相容。目标末维连续，调用前应准备相容的物理布局。Vector 生产的数据跨核供 Cube 消费时需要显式同步。

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

**本轮 950 交付范围（NPU IR 内部转测）：** 同类型 FP16/BF16，4D NZ→2D ND，默认空配置。保留 dma/copy.py、dma/nd2nz.py、dma/nd2nz_window.py 的基础路径；其他类型和批量 5D 路径暂不纳入。AIV 读取前须同步并匹配接收子块。

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

**本轮 950 交付范围（NPU IR 内部转测）：** A5 已有 lowering 和模板，限定 FP16/BF16/FP32→FP32 或 INT32→INT32；n 是常量 1..1024，None 仅在源末维为合法常量时可用。rank 1..4，前导维为 1，末维 stride=1，静态正 strides/offset；容量至少 roundUp(n,16) 个元素，L1 32B/BT 64B 地址对齐且不越界。内部测试使用合法 BT 地址，不承诺 tla.alloc 支持 BT。动态 n、任意二维矩阵、FP16→FP16 BT 不纳入。旧 copy.py 只有 IR 构造证据。

签名：`tla.copy_l1_to_bt(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.l12bt。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L1<br>数据类型：见本轮 950 交付范围 | L12BTOp 校验地址空间，ODS 未枚举元素类型。偏置格式与目标实现有关。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：BT<br>数据类型：见本轮 950 交付范围 | L12BTOp 校验地址空间，ODS 未枚举元素类型。偏置格式与目标实现有关。 | 必填 |
| `params` | 控制搬运的配置 | tla.CopyL12BTParams 或 None | None / tla.CopyL12BTParams 实例 | None 创建默认配置。 | None |
| `params.n` | 传给 L12BTOp 的 N 轴长度 | 编译期 int / None | None 或编译期常量 1..1024 | None 使用 src.shape[-1]。实际读取不能越界。 | None |

共同约束：源为 L1、目的为 BiasBUF。当前 A5 lowering 和模板按上述交付范围校验；旧小测试中的 FP16/FP32 仅是 IR 构造用例。

仓库 IR 构造用例：[tla_ut/dma/copy.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/copy.py#L45)。

<a name="api-tla-copy-l1-to-fb"></a>

## tla.copy_l1_to_fb

**本轮不需要。** 入口直接抛出 NotImplementedError。保留规格供追溯。

签名：`tla.copy_l1_to_fb(src, dst, params=None)`

**预留入口。** 当前调用会抛出 NotImplementedError。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L1<br>数据类型：无可执行签名 | 前端校验空间和配置对象后总是抛 NotImplementedError，不生成 DMA op。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：FB<br>数据类型：无可执行签名 | 前端校验空间和配置对象后总是抛 NotImplementedError，不生成 DMA op。 | 必填 |
| `params` | 控制搬运的配置 | tla.CopyL12FBParams 或 None | None / tla.CopyL12FBParams 实例 | None 创建默认配置。 | None |

<a name="api-tla-copy-l0c-to-ub"></a>

## tla.copy_l0c_to_ub

**本轮 950 交付范围（TA转测）：** 仅 FP32→FP32，dma_mode=NZ2ND、dual_dst_mode=ROW_SPLIT、sub_block_idx=None。其他字段固定默认：NO_QUANT、NO_RELU、quant_scale=None、unit_flag_ctrl/group_id=None；属性缺省不等于 DISABLED。TODO：其他布局、量化、激活、缩放、切分和手动 UnitFlag 另行验证。

签名：`tla.copy_l0c_to_ub(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.fixpipe。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L0C；float32 | 只按本轮组合转测。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：UB；float32 | 与所选布局和转换匹配。 | 必填 |
| `params` | 控制搬运的配置 | tla.FixpipeL0C2UBParams 或 None | None / tla.FixpipeL0C2UBParams 实例 | None 创建默认配置。 | None |
| `params.dma_mode` | 选择输出布局 | 编译期 tla.FixpipeDMAMode | NZ2ND | 本轮固定此模式，调用时显式指定。 | FixpipeDMAMode.NZ2NZ |
| `params.dual_dst_mode` | 选择单目标或双 AIV 子块的切分方式 | 编译期 tla.FixpipeDualDstMode | ROW_SPLIT | 本轮固定行切分。 | FixpipeDualDstMode.NO_DUAL |
| `params.sub_block_idx` | 选择单目标 AIV 子块 | 编译期 tla.FixpipeSubBlock 或 None | None | ROW_SPLIT 必须显式 None。 | FixpipeSubBlock.SUB_BLOCK_0 |

**调用示例**

示例使用 DSA 的 FP32 行切分组合。

```python
@tla.jit
def example(acc_l0c, dst_ub):
    with tla.scope.cube():
        tla.copy_l0c_to_ub(acc_l0c, dst_ub, tla.FixpipeL0C2UBParams(
            dma_mode=tla.FixpipeDMAMode.NZ2ND,
            dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
```

仓库测试：[tla_ut/cube/gemm_bias.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/cube/gemm_bias.py#L35)。


<a name="api-tla-copy-l0c-to-l1"></a>

## tla.copy_l0c_to_l1

**本轮 950 交付范围（NPU IR 内部转测）：** 仅 FP32→FP16、NZ2NZ、F322F16；固定 NO_RELU、quant_scale=None、channel_split=False、c0_pad_en=True、unit_flag_ctrl/group_id=None。对应 dma/fixpipe.py 的 l1_nz。TODO：其他类型、布局、channel_split、关闭 C0 padding、激活和手动 UnitFlag 不纳入本轮。

签名：`tla.copy_l0c_to_l1(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.fixpipe。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L0C；float32 | 只按本轮组合转测。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：L1；float16 | 与所选布局和转换匹配。 | 必填 |
| `params` | 控制搬运的配置 | tla.FixpipeL0C2L1Params 或 None | None / tla.FixpipeL0C2L1Params 实例 | None 创建默认配置。 | None |
| `params.dma_mode` | 选择输出布局 | 编译期 tla.FixpipeDMAMode | NZ2NZ | 本轮固定此模式，调用时显式指定。 | FixpipeDMAMode.NZ2NZ |
| `params.pre_quant` | 选择写出时的类型转换 | 编译期 tla.FixpipePreQuantMode | F322F16 | 依次匹配 FP32 / FP16 / BF16 输出；L1 本轮仅 FP16。 | FixpipePreQuantMode.NO_QUANT |

**调用示例**

目的 L1 使用与默认 NZ→NZ 兼容的布局。

```python
@tla.jit
def example(acc_l0c, dst_l1):
    with tla.scope.cube():
        tla.copy_l0c_to_l1(acc_l0c, dst_l1, tla.FixpipeL0C2L1Params(
            pre_quant=tla.FixpipePreQuantMode.F322F16))
```

仓库测试：[tla_ut/dma/fixpipe.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/fixpipe.py#L69)。


<a name="api-tla-store-l0c-to-gm"></a>

## tla.store_l0c_to_gm

**本轮 950 交付范围（NPU IR 内部转测）：** NZ2ND，FP32→FP32/FP16/BF16 分别使用 NO_QUANT/F322F16/F322BF16；固定 NO_RELU、quant_scale=None、unit_flag_ctrl/group_id=None。对应 dma/fixpipe.py 的 gm_fp32/gm_fp16/gm_bf16。TODO：其他布局、INT32、缩放、激活和手动 UnitFlag 不纳入本轮。

签名：`tla.store_l0c_to_gm(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.fixpipe。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L0C；float32 | 只按本轮组合转测。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：GM；float32 / float16 / bfloat16 | 与所选布局和转换匹配。 | 必填 |
| `params` | 控制搬运的配置 | tla.FixpipeL0C2GMParams 或 None | None / tla.FixpipeL0C2GMParams 实例 | None 创建默认配置。 | None |
| `params.dma_mode` | 选择输出布局 | 编译期 tla.FixpipeDMAMode | NZ2ND | 本轮固定此模式，调用时显式指定。 | FixpipeDMAMode.NZ2NZ |
| `params.pre_quant` | 选择写出时的类型转换 | 编译期 tla.FixpipePreQuantMode | NO_QUANT / F322F16 / F322BF16 | 依次匹配 FP32 / FP16 / BF16 输出；L1 本轮仅 FP16。 | FixpipePreQuantMode.NO_QUANT |

**调用示例**

dst_gm 是已构造的 GM 缓冲区，形状与 NZ→ND 模式相容。

```python
@tla.jit
def example(acc_l0c, dst_gm):
    with tla.scope.cube():
        tla.store_l0c_to_gm(acc_l0c, dst_gm, tla.FixpipeL0C2GMParams(
            dma_mode=tla.FixpipeDMAMode.NZ2ND))
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
