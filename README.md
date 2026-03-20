# rir_sim_SE

一个面向语音增强数据生成的小房间 RIR 仿真项目。

只靠纯实录 RIR，数据量不够，房间和阵列也不够可控；靠纯合成 RIR，又容易太理想，和真实录音之间有落差。
先从真实脉冲响应里提一点房间物理状态和声学参数，再在这个状态附近继续采样，生成一批还能控制、也尽量别太假的 RIR 数据。

## 项目在做什么

现在的主链路只有一条：

1. 从 `configs/rir_sim_se_config.json` 读取房间先验、阵列配置、材料先验和一些生成参数。
2. （1）读取 `acoustic_state.json`/（2）用实录脉冲响应做一次反演，再把结果存成 state。
3. 用 `engine/sound_field_sim/base_engine.py` 生成一条完整 RIR。
4. 在同一条 RIR 上再构造 `ref1` 和 `ref2`，方便后面 SE 任务直接使用。
5. 后续可以再和干声卷积，得到 wet 样本。

不是单纯随机房间，也不是完整几何仿真工具。更像是一个从真实房间里反演可复用声学先验，再往外扩数据的生成器。

## Engine 侧做了什么

物理侧核心文件：[engine/sound_field_sim/base_engine.py](/z:/dataset_rir/rir_sim_SE/engine/sound_field_sim/base_engine.py)。

基础还是 shoebox 房间假设，早期反射主干建立在 `pyroomacoustics` 的 image-source / ISM 思路上。直达声和早期反射部分还是按房间尺寸、源麦位置、墙面材料这些东西来算的。

在这个主干上，engine 另外做了几件事：

- 用材料库给墙、地板、天花板分配频率相关的吸收和散射参数。
- 晚期混响不是单一指数尾巴，而是按频带生成的、多衰减系数的 diffuse tail。
- 低频部分会额外补一个 modal tail，让小房间低频不至于太空。
- 声源侧不再只是简单心形指向，还加了频率相关的人声辐射方向图，以及 source-side 的 head-shadow / torso scattering 近似。
- 最后输出 `rir`、`ref1`、`ref2` 三种版本。`ref1` 更偏“直达加早期”，`ref2` 更稀疏一些，主要保留直接路径和几个早期峰。

这个 engine 还是小房间、shoebox、可控采样这一类问题上的工程化近似，不是 FEM / BEM 那种高保真波动求解器。追求数据生成时的可用性、可控性和速度，不是把每个复杂房间都一比一重建出来。

## Inverse 是基于什么做的

反演入口在 [acoustic_inversion.py](/z:/dataset_rir/rir_sim_SE/acoustic_inversion.py)。

这部分现在是两阶段：

- 第一阶段做统计型反演。从实录脉冲响应里估计 RT60、分频带 RT60、DRR、C50，还有噪声强度和频谱倾斜这些量，然后把这些结果写回 engine，作后续生成先验范围。
- 第二阶段做早期结构分析。从 impulse-like recording 抽取直达声到达时间、前几个早期回声的时延和相对强度、50 ms / 80 ms 内的回声密度，以及分频带 EDT 这一类更贴近房间早期结构的物理量。

 `acoustic_state.json` 不是一个完整的房间几何模型，也不是整个 engine 的序列化快照。保存的是一组后面继续生成时用到的声学先验。

inverse 部分定位：

- 真实房间定标里第一步，把纯 config 房间先验往真实录音方向拉近。
- 不是完整几何反演。房间尺寸先验默认还是来自 config，除非以后真的做 `estimated_room_range` 这一类几何估计。

## 项目基础

底层基础主要有三块：

- `pyroomacoustics` 提供的小房间 ISM 主干。
- 一套围绕小房间 SE 数据生成写的工程化物理补偿，包括材料、晚期混响、低频模态、声源辐射这些部分。
- 一套从实录脉冲响应提取统计参数和早期结构参数的反演逻辑，用来把“纯合成房间”往“更像当前真实房间”这边拉。

不是从零造了一套全新的声学理论，而是在现有 shoebox/ISM 框架上，把 SE 任务真正关心的几个特定点补强了。

## 主要想解决什么困难

语音增强数据准备里很常见的几个问题：

- 实录 RIR 太少，没法大规模扩数据。
- 纯 synthetic RIR 太整齐，模型容易学到不真实的房间分布。
- 想保留一点真实房间特征，但又不想每生成一条都重新做复杂反演。
- SE 任务对直达声、早期反射、晚期尾巴、低频房间效应都比较敏感，简单的随机混响不太够。

所以现在这条链路的思路就是：先做一次轻量反演，拿到一个可复用的 acoustic state，再围绕这个 state 采样很多条不同的 RIR。

## 主要文件

- `main.py`：批量生成入口。
- `config.py`：配置结构和配置加载。
- `acoustic_inversion.py`：把 config 变成 engine，并执行两阶段反演。
- `rir_sim_se.py`：state 的保存/加载，以及 `rir / ref1 / ref2` 的统一生成入口。
- `audio_io.py`：音频读写和卷积。
- `engine/sound_field_sim/base_engine.py`：唯一的物理引擎。

## 运行

```bash
pip install -r requirements.txt
python main.py
```

运行前主要看三处：

- `configs/rir_sim_se_config.json`
- `main.py` 里的 `state_choice`
- `main.py` 里的输出目录

## 测试

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```
