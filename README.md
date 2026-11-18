# snow
demo
# 雪地痕迹系统

> Unity URP · Compute Shader 路径距离场 · RenderTexture 数据流

角色在雪地上行走时留下**真实隆起**的痕迹：中心凹陷、边缘堆雪，
离开后痕迹会被新雪慢慢填平。

核心不是"往贴图上画一个圆"，而是**每帧从整条路径重建一次有符号距离场**，
让痕迹的形状、宽度、边缘都由数学决定，而不是由笔刷叠加决定。

---

## 目录

- [要解决的问题](#要解决的问题)
- [三代方案演进](#三代方案演进)
- [路径距离场方案](#路径距离场方案)
- [数据流](#数据流)
- [痕迹恢复](#痕迹恢复)
- [调试与调参](#调试与调参)
- [代码结构](#代码结构)
- [关键参数](#关键参数)
- [性能](#性能)
- [已知限制](#已知限制)

---

## 要解决的问题

雪地痕迹最直觉的做法是**盖章**：每帧在 RenderTexture 上画一个圆，代表"这里被踩过"。

它有三个绕不过去的问题：

| 问题 | 原因 |
|---|---|
| **雪被瞬间挖掉** | 笔刷一覆盖，值立刻变成 1，看起来像凭空出现一个坑 |
| **边缘有颗粒 / 跳变** | 多个圆叠加后边界互相咬合，宽度不均匀 |
| **痕迹"超前"于身体** | 盖章发生在角色到达之前，痕迹先出现在前方 |

调参数只能减弱这些症状，**不能消除** —— 因为问题出在"用形状叠加表示连续路径"这件事本身。

---

## 三代方案演进

### 一代：RT 盖章（实心块 + Max 混合）

```hlsl
// 笔刷覆盖即 1
amount = max(amount, stamp);
```

**结果**：一下挖掉一个圆，且叠加处发硬。

### 二代：软边 + 相加混合

```hlsl
// 每次只加一点点，被笼罩的像素逐渐压深
amount += softEdge * delta;
```

**结果**：突然挖掘的问题解决了，但**边界仍然是一串圆的包络**，
拐弯处会有明显的波浪边。

### 三代：路径距离场（当前方案）

不再往 RT 上盖形状，而是**每帧把整条路径重新算一遍**：

```
对每个像素：
    找它到整条路径的最短距离 d
    → 用 d 套一个横截面剖面
    → 得到该像素的压实量
```

**结果**：

- 边界是**数学上完美的等宽曲线**，没有圆的包络
- 拐角天然圆滑（SDF 的并集就是圆角连接）
- 宽度、深度、雪堆全部由一个函数控制，可预测

---

## 路径距离场方案

### compute 主循环

```hlsl
// 每个线程 = RT 上的一个像素
float2 worldPos = _AreaMin.xy + uv * _AreaSize;

for (每条路径线段)
{
    float distanceToPath = DistanceToSegment(worldPos, pointA, pointB);
    float profile = SignedCrossSection(distanceToPath, life, worldPos, endFade, depthScale);
    depression = max(depression, profile);   // 取最深的凹陷
    bermDepth  = min(bermDepth,  profile);   // 取最高的隆起
}

_Result[id.xy] = depression + bermDepth;
```

### 有符号横截面（关键设计）

```hlsl
float SignedCrossSection(float distanceToPath, float remainingLife, float2 worldPos, ...)
{
    float halfWidth = _HalfWidth * (1 + edgeJitter);   // 边缘带噪声抖动
    float t = distanceToPath / halfWidth;

    if (t <= 1.0)
    {
        // 带内：中心最深，向边缘 pow 衰减
        v = pow(1.0 - t, _ProfilePow);
    }
    else
    {
        // 带外：隆起（雪被挤到两侧）
        float u = (t - 1.0) / _BermWidth;
        v = -_BermHeight * sin(PI * u) * endFade;
    }

    v *= noiseAndGrain;   // 噪声 + 颗粒扰动
    return v * remainingLife;
}
```

输出是**带符号**的：

| 符号 | 含义 | 视觉效果 |
|---|---|---|
| `> 0` | 雪被压下去 | 痕迹 / 凹陷 |
| `< 0` | 雪被推上来 | 边缘堆雪 / 隆起 |

**这是和"只压一个坑"最大的区别** —— 真实雪地里，踩下去的同时两侧会隆起。

### 边缘与颗粒

```hlsl
float edgeNoise = SampleNoise(worldPos * 0.65 + 31.7);
float edgeJitter = (edgeNoise - 0.5) * 2.0 * _EdgeJitter;   // 边缘不规则

float grainValue = Hash2D(floor(worldPos * _GrainScale));    // 高频颗粒
v *= (1.0 - _GrainAmount + 2.0 * _GrainAmount * grainValue);
```

- `_EdgeJitter` 让边界不是完美圆弧，像被雪粒啃过
- `_GrainAmount` 打散条纹感，避免看出"数学曲线"

### 距离场 → 痕迹强度

```hlsl
// SnowTrailResolve.shader
float bandInner = _BandWidth * (1.0 - _EdgeSoftness);
float bandOuter = _BandWidth;
float trailAmount = 1.0 - smoothstep(bandInner, bandOuter, distanceToPath);
```

这一步把"距离"翻译成"压实强度"，`_EdgeSoftness` 控制边缘过渡宽度。

---

## 数据流

```
玩家移动
   └─► 记录路径点（超过 pointSpacing 才记）
          └─► StructuredBuffer<float4>（xy = 位置, z = 年龄, w = 深度）
                 └─► Compute Shader（每帧重建整张距离场）
                        └─► RenderTexture (RFloat)
                               └─► 雪地材质的 _TrailMap
                                      └─► 顶点位移 + 法线扰动
```

### 区域跟随

距离场只覆盖一块有限区域（默认 10 × 10 米），区域中心跟随玩家移动。
UV 通过 `_TrailMap_ST` 映射到雪地材质，保证痕迹贴在世界的正确位置。

### 路径点管理

```csharp
// 移动超过 pointSpacing 才记录，避免点过密
if (Vector3.Distance(newPos, _prevPosition) < pointSpacing) return;
RecordPoint(newPos);

// 超出容量时丢弃最老的点
if (_activePointCount >= maxPoints) { 滑动窗口 }
```

---

## 痕迹恢复

每个路径点带**年龄**，横截面里乘 `remainingLife`：

```hlsl
float life = 1.0 - max(p0.z, p1.z);   // 两端年龄取大者
return v * life;
```

年龄随时间增长，超过寿命的点被移出路径：

```
痕迹最深 → 逐渐变浅 → 完全消失
```

效果上就是"新雪慢慢把痕迹填回去"，而不是一次性淡出。

---

## 调试与调参

这套系统的参数很多（宽度、深度、雪堆、边缘、颗粒…），所以做了两层可视化：

| 工具 | 用途 |
|---|---|
| **画布预览** | 相机角落贴一张 Quad（Overlay 队列），实时显示距离场 |
| **PNG 导出** | `ExportCanvasPNG()` 把当前距离场导出成图片，离线看形状 |

```csharp
// 预览 Quad 用 Overlay 队列 + ZTest Always，保证不被角色/地形遮挡
```

**没有这个预览板，调参基本靠猜** —— 痕迹是俯视才能看出来的形状，
在第三人称视角下只能看到"脚下有点东西"。

---

## 代码结构

```
SnowDemo/
├── Scripts/
│   └── SnowTrailPathRT.cs          # 主驱动：路径记录 / RT 管理 / 材质绑定 / 预览
├── Compute/
│   └── SnowTrailField.compute      # 距离场重建（每帧全量重算）
├── Shaders/
│   ├── SnowTrailResolve.shader     # 距离场 → 痕迹强度
│   ├── SnowTrailStamp.shader       # 旧版盖章笔刷（保留作对比）
│   ├── SnowTrailFade.shader        # 痕迹衰减
│   ├── SnowTrailPreview.shader     # RFloat → 可视灰度图
│   ├── SnowTrailPreviewQuad.shader # 相机前预览板
│   └── RTVertexColor.shader        # 立即模式绘制用的顶点色 shader
└── Editor/
    └── SetupSnowTrail.cs           # 一键搭场景
```

---

## 关键参数

| 参数 | 说明 |
|---|---|
| `Field Size` | 距离场覆盖的世界边长（米）|
| `Point Spacing` | 移动多远记一个路径点 |
| `Trail Half Width` | 痕迹半宽（米）|
| `Profile Pow` | 横截面形状：1 = 直线坡，2 = 抛物线，3.5 = 窄沟 |
| `Berm Height / Width` | 边缘堆雪的高度与宽度 |
| `Berm Edge Trim` | 路径两端不生成堆雪的留白 |
| `Edge Jitter` | 边缘不规则程度 |
| `Grain Amount / Scale` | 高频颗粒的强度与密度 |
| `Trail Depth` | 写进雪地材质的痕迹深度 |

---

## 性能

| 项 | 说明 |
|---|---|
| 每帧一次 dispatch | 整张距离场全量重算，分辨率可调（默认 512 × 512）|
| 复杂度 | `像素数 × 路径段数`，路径越长越贵 |
| 优化方向 | 只更新脏区域 / 降低分辨率 / 限制路径点数量 |

**全量重算的取舍**：换来的是"边缘永远干净"，代价是路径很长时开销线性增长。
实测数据：待填。

---

## 已知限制

| 限制 | 说明 |
|---|---|
| **只支持一条连续路径** | 多条独立轨迹需要改成多路径数据结构 |
| **路径点上限** | 超出容量后最老的点被丢弃，痕迹会"断头" |
| **平面假设** | 距离场是 2D 的，陡坡上痕迹会拉伸 |
| **单分辨率** | 没有 LOD，大范围精细痕迹吃显存 |
| **依赖雪地 shader 接口** | 需要材质上有 `_TrailMap` / `_TrailIntensity` 之类属性 |

---

