# DMA评审跟踪

[返回接口目录](../README.md)

<a name="dma-001"></a>

## DMA-001 · tla.load_gm_to_l1

[查看规格](../spec/dma.md#api-tla-load-gm-to-l1)

结论：本轮 `params` 仅保留 `enable_nd2nz`，默认 True，与当前固定 ND→NZ 路径一致。False 的无转换语义尚未验证，本轮应拒绝，不能静默忽略。

TODO（后续支持，不纳入本轮测试）：`l2_cache_ctrl`、`enable_dn2nz`。

支持度及交付实现差异：

- [TLA dma.py](https://github.com/triton-lang/triton-ascend/blob/008dff39bdcc908d6706946189b0c53c5257e4d4/third_party/ascend/language/tla/dma.py#L572) 当前固定生成 ND2NZOp，没有 `enable_nd2nz` / `enable_dn2nz`。交付实现需新增前者并检查 True；其他配置字段不再公开。
- 现有 `enable_nz2nd` / `enable_nz2dn` 是反向转换预留名，True 会报错；不能通过改名就视为支持 DN→NZ。
- `l2_cache_ctrl` 当前映射到 `l2_cache_mode`。NPU IR `cfe3feea3da5cbc603cb71221958cbf6e7bb9f95` 的 `bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp:1038` 读取该属性，Cube ND2NZ 模板传给 DMA cache-control 字段。因此它是已有能力的交付裁剪，不是底层尚未实现。
- [现有 nd2nz 用例](https://github.com/triton-lang/triton-ascend/blob/008dff39bdcc908d6706946189b0c53c5257e4d4/third_party/ascend/unittest/tla_ut/dma/nd2nz.py) 覆盖 cache=None/0/1/2/3，包含 IR、编译与精度检查定义；交付测试需移除 cache 参数矩阵，改为默认配置、enable_nd2nz=True 正例及 False 拒绝测试。
- 后续 `enable_dn2nz` 需要明确转置视图、布局与后端映射，并增加精度测试；本次未确认可直接使用的同名 IR 字段。
- 本次仅更新规格和跟踪，未执行设备验证，也未修改 TLA 或 NPU IR 实现。

<a name="dma-004"></a>

## DMA-004 · tla.copy_l1_to_l0

[查看规格](../spec/dma.md#api-tla-copy-l1-to-l0)

- 确认物理 K 分区字段的单位和有效组合，补充按 L0A/L0B 分开的负例。

- `src`、`dst`：确认目标模板的具体元素和布局组合后纳入泛化测试。

- `params.k_part_idx`、`params.k_part_ceil`、`params.k_part_loop`、`params.k_part_actual`：按目标内核的分区协议补充运行范围和字段间关系。

结论：待决。

<a name="dma-005"></a>

## DMA-005 · tla.make_partition_view

[查看规格](../spec/dma.md#api-tla-make-partition-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="dma-006"></a>

## DMA-006 · tla.make_strided_view

[查看规格](../spec/dma.md#api-tla-make-strided-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="dma-007"></a>

## DMA-007 · tla.make_gather_scatter_view

[查看规格](../spec/dma.md#api-tla-make-gather-scatter-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="dma-008"></a>

## DMA-008 · tla.load_view

[查看规格](../spec/dma.md#api-tla-load-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="dma-009"></a>

## DMA-009 · tla.store_view

[查看规格](../spec/dma.md#api-tla-store-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="dma-010"></a>

## DMA-010 · tla.set_pad_value

[查看规格](../spec/dma.md#api-tla-set-pad-value)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="dma-011"></a>

## DMA-011 · tla.set_post_copy

[查看规格](../spec/dma.md#api-tla-set-post-copy)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

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
