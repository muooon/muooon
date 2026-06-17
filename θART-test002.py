import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. パラメータ設定 (さざ波が見えやすいように調整)
# ==========================================
N = 300          # 空間サイズ
STEPS = 250      # タイムステップ数
theta = 1e-5     # 普遍閾値 θ
alpha = 0.48     # 近傍干渉
beta = 0.95      # 過去の記憶ウェイト
window_size = 15 # マクロ場 φ を抽出するウィンドウ

s = np.zeros((STEPS, N), dtype=int)

# ==========================================
# 2. 初期条件: 中央を「サイン波」でしばらく揺らし続ける
# ==========================================
injection_steps = 40
for t in range(injection_steps):
    val = np.sin(2 * np.pi * t / 10)
    if val > 0.3:
        s[t, N//2-2 : N//2+3] = 1
    elif val < -0.3:
        s[t, N//2-2 : N//2+3] = -1

# ==========================================
# 3. θ-ART 更新ループ (修正版)
# ==========================================
for t in range(1, STEPS - 1):
    # 初期注入期間は、注入した中央以外のエリアのみ更新し、それ以降は全空間更新
    if t < injection_steps:
        space_range = list(range(1, N//2-2)) + list(range(N//2+3, N - 1))
    else:
        space_range = range(1, N - 1)
        
    for i in space_range:
        neighbor_effect = alpha * (s[t, i-1] + s[t, i+1])
        inertia = beta * (s[t, i] - s[t-1, i])
        raw_wave = neighbor_effect + inertia
        
        if raw_wave > theta:
            s[t+1, i] = 1
        elif raw_wave < -theta:
            s[t+1, i] = -1
        else:
            s[t+1, i] = 0

# ==========================================
# 4. マクロ場 φ(x, t) の抽出 (移動平均)
# ==========================================
phi = np.zeros_like(s, dtype=float)
for t in range(STEPS):
    phi[t, :] = np.convolve(s[t, :], np.ones(window_size)/window_size, mode='same')

# ==========================================
# 5. 【追加】時間差分・空間差分の計算 (波動方程式との接続)
# ==========================================
# 2階時間差分: d2phi_dt2 = phi(t+1, x) - 2*phi(t, x) + phi(t-1, x)
d2phi_dt2 = phi[2:, 1:-1] - 2 * phi[1:-1, 1:-1] + phi[:-2, 1:-1]

# 2階空間差分: d2phi_dx2 = phi(t, x+1) - 2*phi(t, x) + phi(t, x-1)
d2phi_dx2 = phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2]

# ==========================================
# 6. プロット (4画面構成で比較)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# ① ミクロ場
im0 = axes[0, 0].imshow(s.T, cmap='bwr', aspect='auto', origin='lower')
axes[0, 0].set_title("① Micro Grid $s(x,t)$")
axes[0, 0].set_ylabel("Space (x)")
fig.colorbar(im0, ax=axes[0, 0])

# ② マクロ場
im1 = axes[0, 1].imshow(phi.T, cmap='bwr', aspect='auto', origin='lower')
axes[0, 1].set_title("② Macro Field $\phi(x,t)$")
fig.colorbar(im1, ax=axes[0, 1])

# ③ 時間2階差分
im2 = axes[1, 0].imshow(d2phi_dt2.T, cmap='bwr', aspect='auto', origin='lower')
axes[1, 0].set_title(r"③ Time 2nd Diff: $\Delta_t^2 \phi$")
axes[1, 0].set_xlabel("Time Step (t)")
axes[1, 0].set_ylabel("Space (x)")
fig.colorbar(im2, ax=axes[1, 0])

# ④ 空間2階差分
im3 = axes[1, 1].imshow(d2phi_dx2.T, cmap='bwr', aspect='auto', origin='lower')
axes[1, 1].set_title(r"④ Space 2nd Diff: $\Delta_x^2 \phi$")
axes[1, 1].set_xlabel("Time Step (t)")
fig.colorbar(im3, ax=axes[1, 1])

plt.tight_layout()
plt.show()

# ==========================================
# 7. 散布図による波動方程式の検証 (おまけ)
# ==========================================
plt.figure(figsize=(6, 5))
# データの中心部（波が綺麗に伝播している領域）をサンプリング
sample_t = slice(50, 200)
sample_x = slice(50, 250)
plt.scatter(d2phi_dx2[sample_t, sample_x].flatten(), 
            d2phi_dt2[sample_t, sample_x].flatten(), 
            alpha=0.1, color='purple', s=1)
plt.title("Wave Equation Validation")
plt.xlabel("Space 2nd Derivative ($\Delta_x^2 \phi$)")
plt.ylabel("Time 2nd Derivative ($\Delta_t^2 \phi$)")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()