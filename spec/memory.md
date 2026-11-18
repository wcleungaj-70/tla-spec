# 内存接口规格

[返回接口目录](../README.md)

本轮标有“950 交付范围”的接口按该段限定转测；表内其他历史类型范围不属于本轮承诺。裁剪参数的默认行为仍须在实现内部保持，不能连同其语义删除。

<a name="api-tla-alloc"></a>

## tla.alloc

计划签名：`tla.alloc(shape, dtype, space=None)`

返回：tla.buffer。

分配由 `space` 指定地址空间。多缓冲版本数通过 [tla.annotate_multi_buffer](schedule.md#api-tla-annotate-multi-buffer) 单独标注。当前实现仍接收 `versions` 和 `scope`，签名变更见 [MEMORY-002](../tracking/memory.md#memory-002)。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `shape` | 逻辑形状 | tuple[int, ...] / list[int] | 非空正整数组 | 按元素计。 | 必填 |
| `dtype` | 元素类型 | 编译期 tl.dtype | 可物化为 memref 元素的整数或浮点 dtype | 分配类型须同时满足该空间的内存规划和后续操作的 dtype 约束。 | 必填 |
| `space` | 指定地址空间 | 编译期 tla.AddressSpace / str / None | 地址空间：UB / L1 / L0C（已确认分配路径） | 省略或传 None 时为 UB。GM 禁止 alloc，L0A_SF/L0B_SF 无映射。其他有空间映射的名字不自动获得分配支持。 | None |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.vector():
        tile = tla.alloc((16, 16), tl.float16, space='ub')
```

仓库测试：[tla_ut/scope/pointer_abi.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/pointer_abi.py#L15)。

<a name="api-tla-reshape"></a>

## tla.reshape

签名：`tla.reshape(src, shape)`

返回：tla.buffer。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.buffer | 地址空间及数据类型继承源缓冲区 | 源必须是静态连续 row-major 缓冲区。元素数保持不变。 | 必填 |
| `shape` | 逻辑形状 | tuple[int, ...] / list[int] | 非空正整数组 | reshape 后元素总数相同。 | 必填 |

**调用示例**

x_ub 为连续的 16×16 UB 缓冲区。

```python
@tla.jit
def example(x_ub, y):
    flat = tla.reshape(x_ub, (256,))
    tla.store_ub_to_gm(flat, tla.view(y, (256,)))
```

仓库测试：[tla_ut/vector/subview.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/subview.py#L64)。

<a name="api-tla-view"></a>

## tla.view

签名：`tla.view(src, shape, strides=None, offsets=None, space='gm')`

返回：tla.buffer。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | 标量 Triton 指针 / tla.buffer | 指针源继承指针元素类型；GM 元素必须为整字节宽的整数或浮点 | buffer 源保持 dtype、地址空间和原 pitch。 | 必填 |
| `shape` | 逻辑形状 | tuple[int, ...] / list[int] | 非空正整数组 | 按元素计。 | 必填 |
| `strides` | 各轴步长 | tuple[int, ...] / list[int] / None | 非空正整数组 / None | 与对应视图秩相同。省略时指针源用 row-major，buffer 源用单位相对步长。 | None |
| `offsets` | 基于源地址或视图的元素偏移 | tuple[int / 标量整数 tl.tensor, ...] / None | 按元素计的整数偏移 / None | 维数须匹配。静态子视图越界拒绝，动态偏移由调用者保证。 | None |
| `space` | 指定地址空间 | 编译期 str / tla.AddressSpace | 地址空间：GM / UB / L1 / L0A / L0B / L0C / SSBUF / BT / FB | 指针源指定目标空间。buffer 源要求保留默认参数。 | 'gm' |

**调用示例**

```python
@tla.jit
def example(x, dst_ub):
    tile = tla.view(x, (16, 16), offsets=(0, 0))
    tla.load_gm_to_ub(tile, dst_ub)
```

仓库测试：[tla_ut/scope/pointer_abi.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/pointer_abi.py#L16)。

<a name="api-tla-subview"></a>

## tla.subview

签名：`tla.subview(src, offsets, sizes, strides=None)`

返回：tla.buffer。

编译表示：memref.subview。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.buffer | 地址空间及数据类型继承源缓冲区 | 同秩切片，结果与源共享存储。 | 必填 |
| `offsets` | 基于源地址或视图的元素偏移 | tuple[int / 标量整数 tl.tensor, ...] | 按元素计的整数偏移 | 维数须匹配。静态子视图越界拒绝，动态偏移由调用者保证。 | 必填 |
| `sizes` | 各轴要截取的元素个数 | tuple[int, ...] / list[int] | 非空正整数组 | 按元素计。 | 必填 |
| `strides` | 各轴步长 | tuple[int, ...] / list[int] / None | 非空正整数组 / None | 与对应视图秩相同。省略时指针源用 row-major，buffer 源用单位相对步长。 | None |

**调用示例**

x_ub 为至少 16×16 个元素的二维缓冲区，目的视图与子视图类型一致。

```python
@tla.jit
def example(x_ub, y):
    tile = tla.subview(x_ub, (0, 0), (16, 16))
    tla.store_ub_to_gm(tile, tla.view(y, (16, 16)))
```

仓库测试：[tla_ut/simd/predicate_views.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/predicate_views.py#L39)。

<a name="api-tla-fill"></a>

## tla.fill

**本轮 950 交付范围（NPU IR 内部转测）：** 仅 UB、连续 row-major、offset=0 的 FP32/BF16/INT32（alloc 或合法 reshape）。标量 value 按前端转换规则；排除 bool/i1、向量 value、其他地址空间及带步长/非零偏移 subview。INT32 禁止浮点 value 并检查常量范围。lowering 为标量 hivm.hir.vbrc，参考 memory/fill.py、fill_frontend_contract.py。

签名：`tla.fill(dst, value)`

返回：None。

编译表示：hivm.hir.vbrc。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `dst` | 目标缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：<br>B8：不支持<br>B16：bfloat16<br>B32：int32 / float32<br>B64：不支持 | 零 offset，连续 row-major。 | 必填 |
| `value` | 输入值 | int / float / bool / 标量 tl.tensor | 可转换为 dst 元素类型的标量 | 向全部目标元素广播。 | 必填 |

**调用示例**

out_ub 是符合约束的连续 UB 缓冲区。

```python
@tla.jit
def example(out_ub):
    tla.fill(out_ub, 0)
```

## 范围说明

以下条目在当前整理表中没有可定位的 TLA 接口名。

| 原序号 | 范围结论 | 涉及场景 |
| --- | --- | --- |
| 161 | 类似torch.tensor，workspace 明确不支持 | T.empty和tensor 有关的都疑似global |
| 162 | 类似torch.tensor，workspace 明确不支持 | T.make_tensor 有关的都疑似global |
