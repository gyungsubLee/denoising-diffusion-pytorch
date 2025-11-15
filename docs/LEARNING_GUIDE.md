# Denoising Diffusion Probabilistic Model (DDPM) - 학습 가이드

## 📚 개요

이 프로젝트는 **Denoising Diffusion Probabilistic Models (DDPM)**의 PyTorch 구현입니다. Diffusion 모델은 GAN과 경쟁할 수 있는 새로운 생성 모델링 접근 방식으로, 데이터 분포의 기울기를 추정하기 위해 denoising score matching을 사용하고, 실제 분포에서 샘플링하기 위해 Langevin sampling을 사용합니다.

---

## 📖 주요 참고 논문

### 1. **핵심 논문: DDPM (2020)**
```bibtex
@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    title       = {Denoising Diffusion Probabilistic Models},
    booktitle   = {NeurIPS 2020},
    year        = {2020},
    url         = {https://arxiv.org/abs/2006.11239}
}
```

### 2. **개선된 DDPM (2021)**
- **Improved Denoising Diffusion Probabilistic Models** (Nichol & Dhariwal, 2021)
- Cosine noise schedule 제안
- 학습된 분산(variance) 사용

### 3. **DDIM (2021)**
- **Denoising Diffusion Implicit Models** (Song et al., 2021)
- 빠른 샘플링을 위한 non-Markovian 프로세스
- 250 스텝으로 고품질 샘플 생성 가능

### 4. **기타 중요 논문**
- **Elucidating the Design Space of Diffusion Models** (Karras et al., 2022)
- **Classifier-Free Diffusion Guidance** (Ho, 2022)
- **Min-SNR Weighting Strategy** (Hang et al., 2023)

---

## 🧮 핵심 수식 및 구현

### 1. **Forward Diffusion Process (순방향 확산)**

Forward process는 점진적으로 노이즈를 추가하는 과정입니다.

#### 수식

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \cdot I)$$

여기서:
- $x_t$: 시간 t에서의 noisy 이미지
- $\beta_t$: 시간 t에서의 노이즈 스케줄 (variance schedule)
- $\mathcal{N}$: 가우시안 분포

#### 중요한 성질: Closed-form 샘플링

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1 - \bar{\alpha}_t) \cdot I)$$

여기서 $\bar{\alpha}_t = \prod_{i=1}^t (1 - \beta_i)$ (alpha cumulative product)

#### 재매개변수화 (Reparameterization)

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

#### 코드 구현

```python
# denoising_diffusion_pytorch.py:787-793
def q_sample(self, x_start, t, noise = None):
    noise = default(noise, lambda: torch.randn_like(x_start))

    return (
        extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )
```

**수식 대응:**
- `sqrt_alphas_cumprod` = $\sqrt{\bar{\alpha}_t}$
- `sqrt_one_minus_alphas_cumprod` = $\sqrt{1 - \bar{\alpha}_t}$
- `x_start` = $x_0$
- `noise` = $\epsilon$

---

### 2. **Reverse Diffusion Process (역방향 확산)**

Reverse process는 노이즈로부터 이미지를 생성하는 과정입니다.

#### 수식

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

#### Posterior 분포

원본 이미지 $x_0$를 알 때의 posterior:

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \cdot I)$$

여기서:

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \cdot \beta_t}{1 - \bar{\alpha}_t} \cdot x_0 + \frac{\sqrt{\alpha_t} \cdot (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \cdot x_t$$

$$\tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)} \cdot \beta_t$$

#### 코드 구현

```python
# denoising_diffusion_pytorch.py:646-653
def q_posterior(self, x_start, x_t, t):
    posterior_mean = (
        extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
        extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = extract(self.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
    return posterior_mean, posterior_variance, posterior_log_variance_clipped
```

**초기화 시 계산 (denoising_diffusion_pytorch.py:577-589):**

```python
# β̃_t 계산
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# μ̃_t의 계수들
posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
```

---

### 3. **Training Objective (학습 목표)**

#### 세 가지 학습 목표

이 구현체는 세 가지 학습 목표를 지원합니다:

**1) Noise Prediction (ε-prediction)**

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

**2) x_0 Prediction**

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \|x_0 - \hat{x}_\theta(x_t, t)\|^2 \right]$$

**3) v-Prediction** (Progressive Distillation 논문, Imagen-Video에서 사용)

$$v_t = \sqrt{\bar{\alpha}_t} \cdot \epsilon - \sqrt{1 - \bar{\alpha}_t} \cdot x_0$$

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \|v_t - v_\theta(x_t, t)\|^2 \right]$$

#### 코드 구현

```python
# denoising_diffusion_pytorch.py:795-840
def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
    b, c, h, w = x_start.shape
    noise = default(noise, lambda: torch.randn_like(x_start))

    # Forward process: x_0 → x_t
    x = self.q_sample(x_start = x_start, t = t, noise = noise)

    # 모델 예측
    model_out = self.model(x, t, x_self_cond)

    # Objective에 따라 target 설정
    if self.objective == 'pred_noise':
        target = noise
    elif self.objective == 'pred_x0':
        target = x_start
    elif self.objective == 'pred_v':
        v = self.predict_v(x_start, t, noise)
        target = v

    # MSE Loss
    loss = F.mse_loss(model_out, target, reduction = 'none')
    loss = reduce(loss, 'b ... -> b', 'mean')

    # Loss weighting (Min-SNR)
    loss = loss * extract(self.loss_weight, t, loss.shape)
    return loss.mean()
```

---

### 4. **Noise Schedules (노이즈 스케줄)**

노이즈 스케줄은 확산 과정의 속도를 제어합니다.

#### Linear Schedule (원본 DDPM)

```python
# denoising_diffusion_pytorch.py:462-469
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
```

**수식:**

$$\beta_t = \beta_{\text{start}} + (\beta_{\text{end}} - \beta_{\text{start}}) \cdot \frac{t}{T}$$

#### Cosine Schedule (Improved DDPM)

```python
# denoising_diffusion_pytorch.py:471-481
def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
```

**수식:**

$$\bar{\alpha}_t = \frac{\cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)}{\cos^2\left(\frac{s}{1 + s} \cdot \frac{\pi}{2}\right)}$$

$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$$

#### Sigmoid Schedule

고해상도 이미지(>64x64)에 더 효과적:

```python
# denoising_diffusion_pytorch.py:483-496
def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
```

---

### 5. **Sampling Methods (샘플링 방법)**

#### DDPM Sampling (Ancestral Sampling)

```python
# denoising_diffusion_pytorch.py:700-716
@torch.inference_mode()
def p_sample_loop(self, shape, return_all_timesteps = False):
    img = torch.randn(shape, device = device)

    for t in reversed(range(0, self.num_timesteps)):
        self_cond = x_start if self.self_condition else None
        img, x_start = self.p_sample(img, t, self_cond)

    return self.unnormalize(img)
```

**수식 (p_sample):**

$$
x_{t-1} =
\begin{cases}
\mu_\theta(x_t, t) + \exp(0.5 \cdot \log \sigma^2_\theta(x_t, t)) \cdot z, & \text{if } t > 0 \\
\mu_\theta(x_t, t), & \text{if } t = 0
\end{cases}
$$

여기서 $z \sim \mathcal{N}(0, I)$

#### DDIM Sampling (빠른 샘플링)

```python
# denoising_diffusion_pytorch.py:719-758
@torch.inference_mode()
def ddim_sample(self, shape, return_all_timesteps = False):
    # Accelerated sampling with fewer timesteps
    times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))

    for time, time_next in time_pairs:
        pred_noise, x_start = self.model_predictions(...)

        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
```

**DDIM 수식:**

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \hat{\epsilon}_\theta(x_t, t) + \sigma_t \cdot z$$

여기서:

$$\sigma_t = \eta \cdot \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \cdot \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$$

- $\eta = 0$ 이면 deterministic
- $\eta = 1$ 이면 DDPM과 동일

---

### 6. **Loss Weighting (Min-SNR)**

**Min-SNR Weighting Strategy** (Hang et al., 2023) 적용

#### SNR (Signal-to-Noise Ratio)

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

#### Loss Weight 계산

```python
# denoising_diffusion_pytorch.py:595-611
snr = alphas_cumprod / (1 - alphas_cumprod)

maybe_clipped_snr = snr.clone()
if min_snr_loss_weight:
    maybe_clipped_snr.clamp_(max = min_snr_gamma)  # default: γ=5

if objective == 'pred_noise':
    loss_weight = maybe_clipped_snr / snr
elif objective == 'pred_x0':
    loss_weight = maybe_clipped_snr
elif objective == 'pred_v':
    loss_weight = maybe_clipped_snr / (snr + 1)
```

**수식:**
- Noise prediction: $w_t = \frac{\min(\text{SNR}(t), \gamma)}{\text{SNR}(t)}$
- x_0 prediction: $w_t = \min(\text{SNR}(t), \gamma)$
- v prediction: $w_t = \frac{\min(\text{SNR}(t), \gamma)}{\text{SNR}(t) + 1}$

---

### 7. **x_0 Reconstruction (원본 이미지 복원)**

모델 출력으로부터 원본 이미지 $x_0$를 복원하는 방법:

#### Noise Prediction → x_0

```python
# denoising_diffusion_pytorch.py:622-626
def predict_start_from_noise(self, x_t, t, noise):
    return (
        extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )
```

**수식:**

$$\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \cdot x_t - \sqrt{\frac{1}{\bar{\alpha}_t} - 1} \cdot \hat{\epsilon}_\theta(x_t, t)$$

#### v-Prediction → x_0

```python
# denoising_diffusion_pytorch.py:640-644
def predict_start_from_v(self, x_t, t, v):
    return (
        extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
        extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
    )
```

**수식:**

$$v_t = \sqrt{\bar{\alpha}_t} \cdot \epsilon - \sqrt{1 - \bar{\alpha}_t} \cdot x_0$$

$$\hat{x}_0 = \sqrt{\bar{\alpha}_t} \cdot x_t - \sqrt{1 - \bar{\alpha}_t} \cdot v_\theta(x_t, t)$$

---

## 🏗️ 아키텍처

### U-Net 구조

이 구현은 **U-Net** 아키텍처를 사용합니다:

1. **Time Embedding**: Sinusoidal positional embedding
   ```python
   # denoising_diffusion_pytorch.py:117-130
   class SinusoidalPosEmb(Module):
       def forward(self, t):
           half_dim = self.dim // 2
           emb = math.log(self.theta) / (half_dim - 1)
           emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
           emb = t[:, None] * emb[None, :]
           emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
           return emb
   ```

2. **ResNet Block**: Wide ResNet 스타일
   - Time embedding을 scale & shift로 주입
   - RMSNorm 사용

3. **Attention**: Multi-head self-attention
   - Flash Attention 지원 (`flash_attn=True`)

4. **Up/Downsampling**:
   - Downsample: Pixel unshuffle (2×2 → 1×1, 4× channels)
   - Upsample: Nearest neighbor + Conv2d

---

## 🚀 사용 방법

### 기본 학습

```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # T=1000
    sampling_timesteps = 250,   # DDIM: 250 steps
    objective = 'pred_v',       # v-prediction
    beta_schedule = 'sigmoid',  # sigmoid schedule
    min_snr_loss_weight = True, # Min-SNR weighting
    min_snr_gamma = 5
)

trainer = Trainer(
    diffusion,
    'path/to/images',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,
    gradient_accumulate_every = 2,
    ema_decay = 0.995,
    amp = True,
    calculate_fid = True
)

trainer.train()
```

### 샘플링

```python
# DDIM 샘플링 (빠름)
sampled_images = diffusion.sample(batch_size = 4)

# 모든 timestep 반환
all_timesteps = diffusion.sample(batch_size = 4, return_all_timesteps = True)
```

---

## 🔬 주요 기능

### 1. Self-Conditioning
- 50% 확률로 이전 예측 $\hat{x}_0$를 조건으로 사용
- FID 개선, 학습 시간 25% 증가

### 2. Offset Noise
- 밝기 조절 개선
- `offset_noise_strength = 0.1` 권장

### 3. EMA (Exponential Moving Average)
- 모델 가중치의 이동 평균 유지
- `ema_decay = 0.995`

### 4. Mixed Precision Training
- `amp = True`로 활성화
- 메모리 절약, 학습 속도 향상

---

## 📊 평가 지표

### FID (Fréchet Inception Distance)

```python
trainer = Trainer(
    diffusion,
    ...,
    calculate_fid = True,
    fid_every = 1000  # 1000 스텝마다 FID 계산
)
```

---

## 🎓 학습 팁

1. **시작 설정**:
   - `objective = 'pred_v'`
   - `beta_schedule = 'sigmoid'` (고해상도) 또는 `'cosine'` (저해상도)
   - `min_snr_loss_weight = True`

2. **학습 안정성**:
   - Gradient accumulation 사용
   - EMA decay 0.995-0.9999
   - Learning rate: 8e-5

3. **빠른 샘플링**:
   - DDIM: `sampling_timesteps = 250`
   - `ddim_sampling_eta = 0.0` (deterministic)

4. **메모리 최적화**:
   - Mixed precision (`amp = True`)
   - Gradient checkpointing
   - Flash Attention

---

## 📚 추가 자료

### YouTube 강의
- [Yannic Kilcher](https://www.youtube.com/watch?v=W-O7AZNzbzQ)
- [AI Coffeebreak with Letitia](https://www.youtube.com/watch?v=344w5h24-h8)
- [Outlier](https://www.youtube.com/watch?v=HoKDTa5jHvg)

### 공식 구현
- [TensorFlow 원본](https://github.com/hojonathanho/diffusion)
- [HuggingFace Annotated Code](https://huggingface.co/blog/annotated-diffusion)

---

## 🔑 핵심 개념 요약

| 개념 | 수식 | 코드 위치 |
|------|------|-----------|
| Forward process | $x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon$ | `q_sample()` |
| Reverse process | $p_\theta(x_{t-1}\|x_t)$ | `p_sample()` |
| Loss (noise) | $\|\|\epsilon - \epsilon_\theta(x_t, t)\|\|^2$ | `p_losses()` |
| Loss (v) | $\|\|v - v_\theta(x_t, t)\|\|^2$ | `p_losses()` |
| DDIM | $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \hat{\epsilon}$ | `ddim_sample()` |
| SNR | $\bar{\alpha}_t / (1 - \bar{\alpha}_t)$ | Loss weighting |

---

**Happy Diffusing! 🎨**
