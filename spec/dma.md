# DMA接口规格

[返回接口目录](../README.md)

<a name="api-tla-load-gm-to-l1"></a>

## tla.load_gm_to_l1

本轮交付签名：`tla.load_gm_to_l1(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.nd2nz。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：GM<br>数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 相同 dtype 和逻辑 tile。ND→NZ，普通模板 2D→4D。已注册的批量 3D→5D 路径不含 FP8。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：L1<br>数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 相同 dtype 和逻辑 tile。ND→NZ，普通模板 2D→4D。已注册的批量 3D→5D 路径不含 FP8。 | 必填 |
| `params` | 控制 ND→NZ 转换的配置 | tla.LoadGM2L1Params 或 None | None / 仅含 enable_nd2nz 的配置 | None 等价于 enable_nd2nz=True。 | None |
| `params.enable_nd2nz` | 启用 GM ND→L1 NZ 转换 | 编译期 bool | 本轮仅支持 True | False 的无转换路径未纳入本轮支持，须明确报错；当前源码尚未定义此字段，交付实现需新增。 | True |

**本轮范围：** `params` 只保留 `enable_nd2nz`。现有实现固定生成 `hivm.hir.nd2nz`，并没有可切换的 `enable_nd2nz` 字段；本轮将既有转换行为显式化，不宣称已支持关闭转换。

**TODO：** 后续支持 `l2_cache_ctrl` 和 `enable_dn2nz`。前者当前已有实现和专用测试，但本轮不交付；后者尚未确认直接映射，不能与 NZ→DN 反向转换混淆。支持度和实现差异见 [DMA-001](../tracking/dma.md#dma-001)。

**调用示例**

x 为 ND float16 tile，dst_l1 为匹配的 NZ 缓冲区。

```python
@tla.jit
def example(x, dst_l1):
    with tla.scope.cube():
        tla.load_gm_to_l1(tla.view(x, (16, 16)), dst_l1)
```

现有 ND→NZ 功能测试（不代表新增配置字段已验证）：[tla_ut/dma/nd2nz.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/nd2nz.py#L22)。

<a name="api-tla-load-gm-to-ub"></a>

## tla.load_gm_to_ub

签名：`tla.load_gm_to_ub(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.load。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：GM<br>数据类型：<br>B1：int1<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | LoadOp 要求同 dtype。无 pad_mode 时形状相容。目标模板的 FP8 注册限 RegBase。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：<br>B1：int1<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | LoadOp 要求同 dtype。无 pad_mode 时形状相容。目标模板的 FP8 注册限 RegBase。 | 必填 |
| `params` | 控制搬运的配置 | tla.LoadGM2UBParams 或 None | None / tla.LoadGM2UBParams 实例 | None 创建默认配置。 | None |
| `params.pad_mode` | 选择填充模式 | 编译期 tla.PadMode 或 None | None / PadNull / PadFirstElem / PadValue | None 或 PadNull 时 src/dst 形状相容且不传 pad_value。填充只改变末维，dst 末维不得小于 src。 | None |
| `params.pad_value` | 填充时写入的标量 | int / float / 标量 tl.tensor / None | None 或标量 | PadValue 要求提供此值，其他模式不应提供。转换为 src 的元素类型。 | None |
| `params.left_padding_num` | 左侧填充的元素数 | int / 标量整数 tl.tensor / None | None 或非负整数 | 单位为元素。调用者保证运行时值和目的容量。 | None |
| `params.right_padding_num` | 右侧填充的元素数 | int / 标量整数 tl.tensor / None | None 或非负整数 | 单位为元素。调用者保证运行时值和目的容量。 | None |
| `params.eviction_policy` | L2 驱逐策略提示 | 编译期 tla.EvictionPolicy 或 None | None / EvictNormal / EvictFirst / EvictLast | None 省略属性。 | None |

**调用示例**

目的缓冲区与源视图均为 64 个 float32 元素。

```python
@tla.jit
def example(x, dst_ub):
    tla.load_gm_to_ub(tla.view(x, (64,)), dst_ub)
```

仓库测试：[tla_ut/scope/pointer_abi.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/pointer_abi.py#L16)。

<a name="api-tla-store-ub-to-gm"></a>

## tla.store_ub_to_gm

签名：`tla.store_ub_to_gm(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.store。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：<br>B1：int1<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | StoreOp 要求同 dtype、形状相容。原子模式另有 dtype 限制。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：GM<br>数据类型：<br>B1：int1<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | StoreOp 要求同 dtype、形状相容。原子模式另有 dtype 限制。 | 必填 |
| `params` | 控制搬运的配置 | tla.StoreUB2GMParams 或 None | None / tla.StoreUB2GMParams 实例 | None 创建默认配置。 | None |
| `params.atomic_kind` | 控制该次 store 的原子更新 | 编译期 tla.AtomicKind 或 None | None / NONE / ADD / MAX / MIN（常规 StoreOp 路径） | 其余枚举存在，但不能从枚举推出该目标上的原子写支持。 | None |

**调用示例**

源缓冲区与目的视图均为 64 个 float32 元素。

```python
@tla.jit
def example(src_ub, y):
    tla.store_ub_to_gm(src_ub, tla.view(y, (64,)))
```

仓库测试：[tla_ut/scope/pointer_abi.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/pointer_abi.py#L17)。

<a name="api-tla-copy-l1-to-l0"></a>

## tla.copy_l1_to_l0

签名：`tla.copy_l1_to_l0(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.l12l0。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：L1<br>数据类型：待确认目标支持集合 | L12L0Op 校验地址空间并要求 6 个 i64 属性，ODS 未枚举元素类型。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：L0A / L0B<br>数据类型：待确认目标支持集合 | L12L0Op 校验地址空间并要求 6 个 i64 属性，ODS 未枚举元素类型。 | 必填 |
| `params` | 控制搬运的配置 | tla.CopyL1ToL0Params 或 None | 有效调用须提供字段完整的 tla.CopyL1ToL0Params | 省略时创建的默认配置缺少硬件分块字段，会报错。 | None |
| `params.is_transpose` | 开启 L1 至 L0 搬运的转置形式 | 编译期 bool | False / True | 该模式仍受目标和布局约束。 | False |
| `params.actual_batch` | 原表中的真实 batch 数 | 编译期 int 或 None | None | 非 None 抛 NotImplementedError。 | None |
| `params.k_part_idx` | 当前 K 分段编号 | 编译期 int | 编译期整数；硬件有效范围待确认 | 调用时必填。作为分段编号须与分区方案一致。 | None |
| `params.k_part` | K 分区长度 | 编译期 int | 编译期整数；硬件有效范围待确认 | 调用时必填。长度及单位按 L12L0 分块协议。 | None |
| `params.k_part_ceil` | 对齐后 K 分区长度 | 编译期 int | 编译期整数；硬件有效范围待确认 | 调用时必填。与 k_part 和物理 tile 一致。 | None |
| `params.k_part_loop` | K 分区循环参数 | 编译期 int | 编译期整数；硬件有效范围待确认 | 调用时必填。 | None |
| `params.k_part_actual` | 本分区实际 K 长度 | 编译期 int | 编译期整数；硬件有效范围待确认 | 调用时必填。不得读出有效 tile。 | None |
| `params.m` | 原生 L12L0 的 M 维参数 | 编译期 int | 编译期整数；硬件有效范围待确认 | 调用时必填。须与目标分块匹配。 | None |

共同约束：L12L0Op 将源限定为 L1，目的限定为 L0A 或 L0B。TableGen 中的分段字段是 i64 属性，字段编码宽度不能作为硬件可执行范围。

仓库 IR 构造用例：[tla_ut/dma/copy.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/copy.py#L42)。

<a name="api-tla-make-partition-view"></a>

## tla.make_partition_view

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`make_partition_view( base: tla.buffer (ptr), shape: list[int], strides: list[int], tile_shape: list[int], dim_map: list[int], padding_value: enum<zero,nan,inf,-inf> )-> tla.tensor_view`

<a name="api-tla-make-strided-view"></a>

## tla.make_strided_view

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`make_strided_view( base: tla.buffer (ptr), shape: list[int], strides: list[int], tile_shape: list[int], dim_map: list[int], padding_value: enum<zero,nan,inf,-inf>, traversal_strides: list[int] )-> tla.tensor_view`

<a name="api-tla-make-gather-scatter-view"></a>

## tla.make_gather_scatter_view

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`make_gather_scatter_view( base: tla.buffer (ptr), shape: list[int], strides: list[int], tile_shape: list[int], sparse_dim: list[int], padding_value: enum<zero,nan,inf,-inf> )-> tla.tensor_view`

<a name="api-tla-load-view"></a>

## tla.load_view

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`load_view( src: tla.tensor_view, dst: tla.buffer, tile: list[list[int]] ) 只用于vector kernel`

<a name="api-tla-store-view"></a>

## tla.store_view

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`store_view( src: tla.buffer, dst: tla.tensor_view, tile: list[list[int]] ) 只用于vector kernel`

<a name="api-tla-set-pad-value"></a>

## tla.set_pad_value

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`set_pad_value()`

<a name="api-tla-set-post-copy"></a>

## tla.set_post_copy

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`set_post_copy()`

<a name="api-tla-stride-store"></a>

## tla.stride_store

范围结论：不需要。保留原提案供追溯。

原始提案：`【主推｜结构化带步长访存，不暴露指针】 stride_store(src: tla.buffer, dst: tla.buffer, offset: tla.expr, strides: tuple[tla.expr, ...], numels: tuple[tla.expr, ...]) -> None stride_load(src: tla.buffer, dst: tla.buffer, offset: tla.expr, other: tla.scalar, strides: tuple[tla.expr, ...], numels: tuple[tla.expr, ...]) -> None # rank<=3；numels 是有效元素数（越界预测），非重复次数；仅 AIV 【SIMD 级｜VF 内逐寄存器写】 simd.store_with_stride(dst: tla.buffer, indices, stride: int, mask: tla.mask, val: tla.vreg) -> None # stride 必须编译期常量且为 32B 块整数倍；无 post-update 【流式推进｜用循环携带地址，不提供可变指针对象】 for addr in tla.carry(init_addr): # 下沉 scf.for iter_args buf = tla.make_tensor(addr, ...) addr = addr + step`

<a name="api-tla-stride-load"></a>

## tla.stride_load

范围结论：不需要。保留原提案供追溯。

原始提案：`【主推｜结构化带步长访存，不暴露指针】 stride_store(src: tla.buffer, dst: tla.buffer, offset: tla.expr, strides: tuple[tla.expr, ...], numels: tuple[tla.expr, ...]) -> None stride_load(src: tla.buffer, dst: tla.buffer, offset: tla.expr, other: tla.scalar, strides: tuple[tla.expr, ...], numels: tuple[tla.expr, ...]) -> None # rank<=3；numels 是有效元素数（越界预测），非重复次数；仅 AIV 【SIMD 级｜VF 内逐寄存器写】 simd.store_with_stride(dst: tla.buffer, indices, stride: int, mask: tla.mask, val: tla.vreg) -> None # stride 必须编译期常量且为 32B 块整数倍；无 post-update 【流式推进｜用循环携带地址，不提供可变指针对象】 for addr in tla.carry(init_addr): # 下沉 scf.for iter_args buf = tla.make_tensor(addr, ...) addr = addr + step`
