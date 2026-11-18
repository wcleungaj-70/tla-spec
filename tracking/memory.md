# 内存评审跟踪

[返回接口目录](../README.md)

<a name="memory-001"></a>

## MEMORY-001 · tla.alloc

[查看规格](../spec/memory.md#api-tla-alloc)

- 原表序号 2 要求分配独占并禁止 planmem retry 复用。当前物理内存是否复用由内存规划决定，需确认该场景的要求。

结论：待决。移除分配参数和物理内存能否复用是两个问题。现有实现仍由后端内存规划决定复用，已确认的签名变更记录在 [MEMORY-002](#memory-002)。

<a name="memory-002"></a>

## MEMORY-002 · tla.alloc 签名调整

[查看规格](../spec/memory.md#api-tla-alloc)

- 目标签名为 `tla.alloc(shape, dtype, space=None)`。删除 `scope` 关键字，调用方用 `space` 指定地址空间。
- 移除 `versions`。需要多缓冲时改用 [tla.annotate_multi_buffer](../spec/schedule.md#api-tla-annotate-multi-buffer)。

结论：设计已确认，待实现。当前签名仍接收这两个参数。实现更新后核对旧调用的迁移、新签名和默认 UB 行为。

<a name="ricky-scope"></a>

## Ricky 接口：950 转测范围依据

本次为源码审查，未执行设备精度测试。TA 提交 `008dff39bdcc908d6706946189b0c53c5257e4d4`，NPU IR 提交 `cfe3feea3da5cbc603cb71221958cbf6e7bb9f95`。工作区有既存改动，相关 DMA 前端和后端模板与提交核对；本次不修改实现。进度仅维护 README，裁剪范围见规格页。

- `tla.fill`：仅 UB、连续 row-major、offset=0 的 FP32/BF16/INT32（alloc 或合法 reshape）。标量 value 按前端转换规则；排除 bool/i1、向量 value、其他地址空间及带步长/非零偏移 subview。INT32 禁止浮点 value 并检查常量范围。lowering 为标量 hivm.hir.vbrc，参考 memory/fill.py、fill_frontend_contract.py。

源码与用例证据（测试定义不代表本轮已通过）：

- [memory/fill.py](https://github.com/triton-lang/triton-ascend/blob/008dff39bdcc908d6706946189b0c53c5257e4d4/third_party/ascend/unittest/tla_ut/memory/fill.py)
- [memory/fill_frontend_contract.py](https://github.com/triton-lang/triton-ascend/blob/008dff39bdcc908d6706946189b0c53c5257e4d4/third_party/ascend/unittest/tla_ut/memory/fill_frontend_contract.py)
