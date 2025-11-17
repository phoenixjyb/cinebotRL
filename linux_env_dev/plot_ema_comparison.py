"""Generate a demo plot comparing EMA smoothing with different alpha values.

Saves: linux_env_dev/plots/ema_comparison.png

Run:
    python linux_env_dev/plot_ema_comparison.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def ema_filter(signal: np.ndarray, alpha: float) -> np.ndarray:
    """Simple exponential moving average filter.

    y[0] = x[0]
    y[t] = alpha * x[t] + (1-alpha) * y[t-1]
    """
    out = np.empty_like(signal, dtype=float)
    if signal.size == 0:
        return out
    out[0] = float(signal[0])
    one_minus = 1.0 - float(alpha)
    for i in range(1, signal.shape[0]):
        out[i] = float(alpha) * float(signal[i]) + one_minus * out[i - 1]
    return out


def make_noisy_signal(n=500, seed=0):
    rng = np.random.default_rng(int(seed))
    t = np.linspace(0.0, 10.0, n)
    # base signal: sum of low-frequency components
    base = 0.5 * np.sin(0.8 * t) + 0.2 * np.cos(0.2 * t) + 0.02 * t
    # add high-frequency jitter
    noise = rng.normal(0.0, 1.0, size=n)
    noisy = base + noise
    return t, base, noisy


def plot_ema_comparison(out_path=None, alphas=None, n=500):
    if alphas is None:
        alphas = [0.02, 0.05, 0.1, 0.2, 0.4]
    t, base, noisy = make_noisy_signal(n=n, seed=42)

    # compute filtered signals
    filtered = {}
    for a in alphas:
        filtered[a] = ema_filter(noisy, a)

    # ensure output dir
    out_dir = os.path.join(os.getcwd(), 'linux_env_dev', 'plots')
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_path or os.path.join(out_dir, 'ema_comparison.png')

    # figure: top = raw vs base, bottom = filtered comparisons
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    ax1.plot(t, noisy, color='0.6', label='noisy (input)', linewidth=0.8)
    ax1.plot(t, base, color='tab:blue', label='true / base', linewidth=1.2)
    ax1.set_ylabel('signal')
    ax1.set_title('Noisy input signal and true base')
    ax1.legend(loc='upper right')

    # plot filtered lines
    cmap = plt.get_cmap('tab10')
    for i, a in enumerate(alphas):
        color = cmap(i % 10)
        ax2.plot(t, filtered[a], label=f'EMA alpha={a}', color=color, linewidth=1.2)
    # overlay noisy faded
    ax2.plot(t, noisy, color='0.85', linewidth=0.6, label='noisy (input)')
    ax2.set_ylabel('filtered')
    ax2.set_xlabel('time')
    ax2.set_title('EMA filtered signals with different alpha')
    ax2.legend(loc='upper right', ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved EMA comparison plot to: {out_path}")
    return out_path


if __name__ == '__main__':
    plot_ema_comparison()
