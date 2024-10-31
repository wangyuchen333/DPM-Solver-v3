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
| "x_start": 数据预测模型 $x_\theta$       | $E_{x_0,\epsilon,t}\left[\omega_2(t)\|\|x_\theta(x_t,t)-x_0\|\|_2^2\right]$ | [DALL·E 2](https://arxiv.org/abs/2204.06125)                  |
| "v": 速度预测模型 $v_\theta$         | $E_{x_0,\epsilon,t}\left[\omega_3(t)\|\|v_\theta(x_t,t)-(\alpha_t\epsilon - \sigma_t x_0)\|\|_2^2\right]$ | [Imagen Video](https://arxiv.org/abs/2210.02303)              |
| "score": 边际分数函数 $s_\theta$       | $E_{x_0,\epsilon,t}\left[\omega_4(t)\|\|\sigma_t s_\theta(x_t,t)+\epsilon\|\|_2^2\right]$ | [ScoreSDE](https://arxiv.org/abs/2011.13456)                  |

#### 采样类型
我们支持以下三种类型的扩散模型采样。您可以通过函数`model_wrapper`中的参数`guidance_type`设置。

| 采样类型                                     | 噪声预测模型的方程                          | 示例论文                                                |
| ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| "uncond": 无条件采样                  | $\tilde\epsilon_\theta(x_t,t)=\epsilon_\theta(x_t,t)$        | [DDPM](https://arxiv.org/abs/2006.11239)                      |
| "classifier": 分类器引导                 | $\tilde\epsilon_\theta(x_t,t,c)=\epsilon_\theta(x_t,t)-s\cdot\sigma_t\nabla_{x_t}\log q_\phi(x_t,t,c)$ | [ADM](https://arxiv.org/abs/2105.05233),  [GLIDE](https://arxiv.org/abs/2112.10741)  |
| "classifier-free": 无分类器引导 (CFG) | $\tilde\epsilon_\theta(x_t,t,c)=s\cdot \epsilon_\theta(x_t,t,c)+(1-s)\cdot\epsilon_\theta(x_t,t)$ | [DALL·E 2](https://arxiv.org/abs/2204.06125),  [Imagen](https://arxiv.org/abs/2205.11487),  [Stable-Diffusion](https://github.com/CompVis/stable-diffusion)  |

#### 2.1. 无引导采样（无条件）
给定的`model`具有以下格式：
```python
model(x_t, t_input, **model_kwargs) -> 噪声 | x_start | v | score
```

我们通过以下方式包装模型：

```python
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type=model_type,  # "噪声"或"x_start"或"v"或"score"
    model_kwargs=model_kwargs,  # 模型的其他输入
)
```

#### 2.2. 有分类器引导采样（条件）
给定的`model`具有以下格式：
```python
model(x_t, t_input, **model_kwargs) -> 噪声 | x_start | v | score
```

对于具有分类器引导的DPMs，我们还需要将模型输出与分类器梯度结合起来。我们需要指定分类器函数和引导比例。分类器函数具有以下格式：
```python
classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
```

其中`t_input`与原始扩散模型`model`中的时间标签相同，`cond`是条件（例如类标签）。

我们通过以下方式包装模型：

```python
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type=model_type,  # "噪声"或"x_start"或"v"或"score"
    model_kwargs=model_kwargs,  # 模型的其他输入
    guidance_type="classifier",
    condition=condition,  # 分类器的条件输入
    guidance_scale=guidance_scale,  # 分类器引导比例
    classifier_fn=classifier,
    classifier_kwargs=classifier_kwargs,  # 分类器函数的其他输入
)
```

#### 2.3. 无分类器引导采样（条件）
给定的`model`具有以下格式：
```python
model(x_t, t_input, cond, **model_kwargs) -> 噪声 | x_start | v | score
```

请注意，对于无分类器引导，模型需要另一个输入`cond`（例如文本提示）。如果`cond`是一个特殊的变量`unconditional_condition`（例如空文本`""`），则模型输出是无条件DPM输出。

我们通过以下方式包装模型：

```python
model_fn = model_wrapper(
    model,
    noise_schedule,
    model_type=model_type,  # "噪声"或"x_start"或"v"或"score"
    model_kwargs=model_kwargs,  # 模型的其他输入
    guidance_type="classifier-free",
    condition=condition,  # 条件输入
    unconditional_condition=unconditional_condition,  # 无条件模型的特殊无条件条件变量
    guidance_scale=guidance_scale,  # 无分类器引导比例
)
```

### 3. 定义DPM-Solver-v3并采样

在定义了`noise_schedule`和`model_fn`之后，我们可以进一步使用它们来定义DPM-Solver-v3并生成样本。

首先，我们定义DPM-Solver-v3实例`dpm_solver_v3`，它将自动处理一些必要的预处理。

```python
dpm_solver_v3 = DPM_Solver_v3(
    statistics_dir="statistics/sd-v1-4/7.5_250_1024", 
    noise_schedule, 
    steps=10, 
    t_start=1.0, 
    t_end=1e-3, 
    skip_type="time_uniform", 
    degenerated=False,
)
```

- `statistics_dir`：存储计算出的EMS（即论文中的三个系数 $l,s,b$）的目录。论文中使用的EMS可以在 https://drive.google.com/drive/folders/1sWq-htX9c3Xdajmo1BG-QvkbaeVtJqaq 获得。对于您自己的模型，您需要按照论文的*附录C.1.1*计算它们的EMS。
- `steps`：采样的步数。由于DPM-Solver-v3使用多步方法，总函数评估次数（NFE）等于`steps`。
- `t_start`和`t_end`：我们从时间`t_start`采样到时间`t_end`。
  - 对于离散时间DPMs，我们不需要指定`t_start`和`t_end`。默认设置是从离散时间标签$N-1$采样到离散时间标签$0$。
  - 对于连续时间DPMs（VP），我们从`t_start=1.0`（默认设置）采样到`t_end`。我们建议`steps <= 15`时`t_end=1e-3`，`steps > 15`时`t_end=1e-4`。对于连续时间DPMs（EDM），我们可以按照[EDM](https://github.com/NVlabs/edm)的训练设置（即`t_start=80.0`和`t_end=0.002`）。

- `skip_type`：采样的时间步长计划（即我们如何从`t_start`到`t_end`的时间离散化）。我们支持4种`skip_type`：
  - `logSNR`：时间步长的均匀对数信噪比。**推荐用于低分辨率图像**。
  - `time_uniform`：时间步长的均匀时间。**推荐用于高分辨率图像**。

  - `time_quadratic`：时间步长的二次时间。

  - `edm`：几何信噪比，由[EDM](https://github.com/NVlabs/edm)提出。在我们的实验中，我们发现它不如均匀对数信噪比或均匀时间好。

- `degenerated`：将EMS退化为次优选择 $l=1,s=0,b=0$，对应于DPM-Solver++。详情见论文的*附录A.2*。

然后我们可以使用`dpm_solver_v3.sample`快速从DPMs采样。这个函数通过DPM-Solver-v3计算时间`t_end`处的ODE解，给定初始`x_T`在时间`t_start`。

```python
x_sample = dpm_solver_v3.sample(
    x_T,
    model_fn,
    order=3,
    p_pseudo=False,
    use_corrector=True,
    c_pseudo=True,
    lower_order_final=True,
    half=False,
)
```

- `order`：DPM-Solver-v3的阶数。我们建议无条件采样使用`order=3`，条件采样使用`order=2`。

- `p_pseudo`：是否使用伪阶预测

