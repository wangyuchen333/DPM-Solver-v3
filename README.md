# DPM-Solver-v3
这个仓库是论文 [DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics](https://openreview.net/forum?id=9fWKExmKa0)（NeurIPS 2023）的官方代码。

<h3><a href="https://ml.cs.tsinghua.edu.cn/dpmv3/">项目页面</a> | <a href="https://arxiv.org/pdf/2310.13268.pdf">论文</a> | <a href="https://arxiv.org/abs/2310.13268">arXiv</a></h3>

DPM-Solver-v3 是一个无需训练的ODE求解器，专门用于快速采样扩散模型，配备了预计算的*实证模型统计（EMS）*来将收敛速度提高高达40%。DPM-Solver-v3在少步骤采样（5~10步）中特别显著且非平凡地提高了质量。

*详情请参考论文和项目页面。*

## 代码示例

我们将DPM-Solver-v3集成到不同的代码库中，之前最先进的采样器 [DPM-Solver++](https://github.com/LuChengTHU/dpm-solver) 和 [UniPC](https://github.com/wl-zhao/UniPC) 也包括在内，方便进行基准测试和比较。

我们将代码示例放在 `codebases/` 中。论文中报告的实验结果可以通过它们复现。

|       名称       |                    原始仓库                    |               预训练模型               |            数据集            |                    类型                    |
| :--------------: | :------------------------------------------------: | :-------------------------------------------: | :---------------------------: | :----------------------------------------: |
|    score_sde     |        https://github.com/yang-song/score_sde_pytorch         | `cifar10_ddpmpp_deep_continuous-checkpoint_8` |           CIFAR-10            |            无条件/像素空间             |
|       edm        |                https://github.com/NVlabs/edm                  |         `edm-cifar10-32x32-uncond-vp`         |           CIFAR-10            |            无条件/像素空间             |
| guided-diffusion |          https://github.com/openai/guided-diffusion           |    `256x256_diffusion/256x256_classifier`     |         ImageNet-256          |              条件/像素空间              |
| stable-diffusion | https://github.com/CompVis/latent-diffusion<br  />https://github.com/CompVis/stable-diffusion  |      `lsun_beds256-model`<br />`sd-v1-4`      | LSUN-Bedroom<br />MS-COCO2014 | 无条件/潜在空间<br />条件/潜在空间 |

## 文档

我们在一个单独的文件 `dpm_solver_v3.py` 中提供了DPM-Solver-v3的PyTorch实现。我们建议参考代码示例来了解它在不同设置中的实用用法。

要使用DPM-Solver-v3，可以按照以下步骤操作。特别感谢 [DPM-Solver](https://github.com/LuChengTHU/dpm-solver) 提供了统一的模型包装器来支持各种扩散模型。

### 1. 定义噪声时间表

*噪声时间表* $\alpha_t,\sigma_t$ 定义了从时间 $0$ 到时间 $t$ 的前向转换核：
$$
p(x_t|x_0)=\mathcal N(x_t;\alpha_tx_0,\sigma_t^2I)
$$
或者等价地
$$
x_t=\alpha_tx_0+\sigma_t\epsilon,\quad \epsilon\sim\mathcal N(0,I)
$$
我们支持两类主要的噪声时间表：

|                名称                 |    Python类    |        定义         |        类型         |
| :---------------------------------: | :----------------: | :------------------------: | :-----------------: |
|      Variance Preserving (VP)       | `NoiseScheduleVP`  | $\alpha_t^2+\sigma_t^2=1$ | 离散/连续        |
| EDM (https://github.com/NVlabs/edm)  | `NoiseScheduleEDM` |  $\alpha_t=1,\sigma_t=t$  |    连续       |

#### 1.1. 离散时间DPMs

##### VP

我们在 `NoiseScheduleVP` 类中支持 $\log\alpha_{t}$ 的分段线性插值，以将离散噪声时间表转换为连续噪声时间表。

我们需要 $\beta_i$ 数组或 $\bar{\alpha}_i$ 数组中的任何一个（详情参见 [DDPM](https://arxiv.org/abs/2006.11239)）来定义噪声时间表。详细关系如下：
$$
\bar{\alpha}_i = \prod (1 - \beta_k)
$$

$$
\alpha_{t_i} = \sqrt{\bar{\alpha}_i}
$$

通过 $\beta_i$ 数组定义离散时间噪声时间表：

```python
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
```

或通过 $\bar{\alpha}_i$ 数组定义离散时间噪声时间表：
```python
noise_schedule = NoiseScheduleVP(schedule='discrete', alphas_cumprod=alphas_cumprod)
```

#### 1.2. 连续时间DPMs

##### VP

我们支持连续时间DPMs的线性时间表和余弦时间表。

|    名称     |                          $\alpha_t$                          | 示例论文                                                |
| :---------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| VP (linear) |  $e^{-\frac{1}{4}(\beta_1-\beta_0)t^2-\frac{1}{2}\beta_0t}$  | [DDPM](https://arxiv.org/abs/2006.11239),[ScoreSDE](https://arxiv.org/abs/2011.13456)  |
| VP (cosine) | $\frac{f(t)}{f(0)}$ ($f(t)=\cos\left(\frac{t+s}{1+s}\frac{\pi}{2}\right)$) | [improved-DDPM](https://arxiv.org/abs/2102.09672)              |

定义连续时间线性噪声时间表，$\beta_0=0.1,\beta_1=20$：
```python
noise_schedule = NoiseScheduleVP(schedule='linear', continuous_beta_0=0.1, continuous_beta_1=20.)
```

定义连续时间余弦噪声时间表，$s=0.008$：
```python
noise_schedule = NoiseScheduleVP(schedule='cosine')
```

##### EDM

```python
noise_schedule = NoiseScheduleEDM()
```

### 2. 定义模型包装器

对于给定的扩散`model`，输入时间为标签（可以是离散时间标签（例如0到999）或连续时间时间（例如0到1）），模型的输出类型可能是“噪声”或“x_start”或“v”或“score”（见`模型类型`），我们将模型函数包装成以下格式：

```python
model_fn(x, t_continuous) -> 噪声
```

其中`t_continuous`是连续时间标签（例如0到1），模型的输出类型是“噪声”，即噪声预测模型。包装的连续时间噪声预测模型`model_fn`用于DPM-Solver-v3。

请注意，DPM-Solver-v3只需要噪声预测模型（即 $\epsilon_\theta(x_t, t)$ 模型，也称为“均值”模型），因此对于预测“均值”和“方差”的扩散模型（例如 [improved-DDPM](https://arxiv.org/abs/2102.09672)），您需要首先自己定义另一个函数，只输出“均值”。

#### 模型类型
我们支持以下四种类型的扩散模型。您可以通过函数`model_wrapper`中的参数`model_type`设置模型类型。

| 模型类型                                        | 训练目标                                           | 示例论文                                                      |
| ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| "噪声": 噪声预测模型 $\epsilon_\theta$ | $E_{x_{0},\epsilon,t}\left[\omega_1(t)\|\|\epsilon_\theta(x_t,t)-\epsilon\|\|_2^2\right]$ | [DDPM](https://arxiv.org/abs/2006.11239),  [Stable-Diffusion](https://github.com/CompVis/stable-diffusion)  |
| "x_start": 数据预测模型 $x_\theta$       | $E_{x_0,\epsilon,t}\left[\omega_2(t)\|\|x_\theta(x_t,t)-x_0\|\|_2^2\right]$ | [DALL·E
