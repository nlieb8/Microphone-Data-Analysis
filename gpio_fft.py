"""
gpio_fft.py
-----------
Runs a Fast Fourier Transform on the state history stored in a GPIOHeapBuffer.

Because GPIO samples may not be perfectly evenly spaced (clock drift, OS jitter),
this module resamples the signal onto a uniform time grid before applying the FFT.

Output:
    FFTResult  -- dataclass holding frequencies, magnitudes, dominant frequency,
                  and the resampled signal used as FFT input.

Usage:
    from gpio_fft import analyse
    from gpio_logger import GPIOLogger

    logger = GPIOLogger(pin=17, freq=100, max_entries=1024)
    logger.start()                      # Ctrl+C to stop

    result = analyse(logger.buffer)
    print(result.dominant_frequency_hz)
    result.print_summary()
"""

import math
from dataclasses import dataclass, field

try:
    import numpy as np
except ImportError:
    raise SystemExit("numpy is required: pip install numpy")

from gpio_logger import GPIOHeapBuffer


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FFTResult:
    """
    Holds the output of an FFT analysis on a GPIO buffer.

    Attributes:
        frequencies_hz      -- array of frequency bins (Hz)
        magnitudes          -- array of FFT magnitudes (same length as frequencies_hz)
        dominant_frequency_hz -- frequency bin with the highest magnitude (excluding DC)
        dominant_magnitude  -- magnitude at the dominant frequency
        sample_rate_hz      -- uniform sample rate used after resampling
        n_samples           -- number of samples fed into the FFT
        resampled_signal    -- the uniform-grid signal the FFT was computed on
        resampled_times     -- timestamps of the uniform grid
    """
    frequencies_hz: np.ndarray
    magnitudes: np.ndarray
    dominant_frequency_hz: float
    dominant_magnitude: float
    sample_rate_hz: float
    n_samples: int
    resampled_signal: np.ndarray
    resampled_times: np.ndarray

    def top_n(self, n: int = 5) -> list[tuple[float, float]]:
        """Return the top-n (frequency_hz, magnitude) pairs, excluding DC (0 Hz)."""
        # Exclude DC bin (index 0)
        idx = np.argsort(self.magnitudes[1:])[::-1][:n] + 1
        return [(float(self.frequencies_hz[i]), float(self.magnitudes[i])) for i in idx]

    def print_summary(self) -> None:
        print(f"\n{'─' * 50}")
        print(f"  GPIO FFT Summary")
        print(f"{'─' * 50}")
        print(f"  Samples analysed : {self.n_samples}")
        print(f"  Sample rate      : {self.sample_rate_hz:.2f} Hz")
        print(f"  Freq resolution  : {self.sample_rate_hz / self.n_samples:.4f} Hz/bin")
        print(f"  Dominant freq    : {self.dominant_frequency_hz:.4f} Hz  "
              f"(magnitude {self.dominant_magnitude:.4f})")
        print(f"\n  Top 5 frequencies:")
        for rank, (freq, mag) in enumerate(self.top_n(5), 1):
            bar = "█" * max(1, int(mag / (self.dominant_magnitude or 1) * 20))
            print(f"    {rank}. {freq:8.4f} Hz  |{bar:<20}|  {mag:.4f}")
        print(f"{'─' * 50}\n")


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyse(
    buffer: GPIOHeapBuffer,
    target_sample_rate_hz: float | None = None,
    window: str = "hann",
) -> FFTResult:
    """
    Run an FFT on the state history in a GPIOHeapBuffer.

    Steps:
      1. Extract sorted (timestamp, state) samples from the heap.
      2. Resample onto a uniform time grid (removes jitter).
      3. Apply a window function to reduce spectral leakage.
      4. Compute the real FFT and return an FFTResult.

    Args:
        buffer               -- a GPIOHeapBuffer (from gpio_logger.py)
        target_sample_rate_hz -- resample to this rate before FFT.
                                 Defaults to the median sample rate of the buffer.
        window               -- window function: 'hann' (default), 'hamming',
                                 'blackman', or 'none'.

    Returns:
        FFTResult

    Raises:
        ValueError  if the buffer has fewer than 4 samples.
    """
    samples = buffer.to_sorted_list()   # sorted by timestamp, oldest first

    if len(samples) < 4:
        raise ValueError(
            f"Buffer has only {len(samples)} sample(s); need at least 4 for FFT."
        )

    times  = np.array([t for t, _ in samples], dtype=np.float64)
    states = np.array([s for _, s in samples], dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. Determine resample rate
    # ------------------------------------------------------------------
    intervals = np.diff(times)
    median_interval = float(np.median(intervals))
    measured_rate   = 1.0 / median_interval

    fs = target_sample_rate_hz if target_sample_rate_hz is not None else measured_rate

    # ------------------------------------------------------------------
    # 2. Resample onto a uniform grid via linear interpolation
    #    (GPIO state is binary, but linear interp gives the cleanest
    #     frequency-domain result; for strict binary you can round after)
    # ------------------------------------------------------------------
    t_start, t_end = times[0], times[-1]
    n_uniform = max(4, int(round((t_end - t_start) * fs)))
    t_uniform = np.linspace(t_start, t_end, n_uniform)
    s_uniform = np.interp(t_uniform, times, states)

    # ------------------------------------------------------------------
    # 3. Apply window to reduce spectral leakage
    # ------------------------------------------------------------------
    win_map = {
        "hann":     np.hanning,
        "hamming":  np.hamming,
        "blackman": np.blackman,
        "none":     lambda n: np.ones(n),
    }
    if window not in win_map:
        raise ValueError(f"Unknown window '{window}'. Choose from: {list(win_map)}")

    win = win_map[window](n_uniform)
    s_windowed = s_uniform * win

    # ------------------------------------------------------------------
    # 4. Real FFT
    # ------------------------------------------------------------------
    fft_vals  = np.fft.rfft(s_windowed)
    freqs     = np.fft.rfftfreq(n_uniform, d=1.0 / fs)
    magnitudes = np.abs(fft_vals) / n_uniform   # normalise by sample count

    # Find dominant frequency, excluding DC (bin 0)
    if len(magnitudes) > 1:
        dominant_idx = int(np.argmax(magnitudes[1:])) + 1
    else:
        dominant_idx = 0

    return FFTResult(
        frequencies_hz=freqs,
        magnitudes=magnitudes,
        dominant_frequency_hz=float(freqs[dominant_idx]),
        dominant_magnitude=float(magnitudes[dominant_idx]),
        sample_rate_hz=fs,
        n_samples=n_uniform,
        resampled_signal=s_uniform,
        resampled_times=t_uniform,
    )


# ---------------------------------------------------------------------------
# Optional: plot with matplotlib
# ---------------------------------------------------------------------------

def plot(result: FFTResult, max_freq_hz: float | None = None) -> None:
    """
    Plot the FFT magnitude spectrum using matplotlib (if available).

    Args:
        result      -- FFTResult from analyse()
        max_freq_hz -- clip x-axis to this frequency (default: Nyquist / 2)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[gpio_fft] matplotlib not installed — skipping plot. pip install matplotlib")
        return

    freqs = result.frequencies_hz
    mags  = result.magnitudes

    if max_freq_hz is not None:
        mask  = freqs <= max_freq_hz
        freqs = freqs[mask]
        mags  = mags[mask]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("GPIO FFT Analysis", fontsize=13, fontweight="bold")

    # Top: resampled time-domain signal
    axes[0].plot(result.resampled_times - result.resampled_times[0],
                 result.resampled_signal, lw=0.8, color="steelblue")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("State (0/1)")
    axes[0].set_title("Resampled GPIO signal")
    axes[0].set_yticks([0, 1])
    axes[0].grid(True, alpha=0.3)

    # Bottom: frequency spectrum
    axes[1].plot(freqs, mags, lw=0.9, color="darkorange")
    axes[1].axvline(result.dominant_frequency_hz, color="crimson",
                    linestyle="--", lw=1.2,
                    label=f"Dominant: {result.dominant_frequency_hz:.4f} Hz")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude (normalised)")
    axes[1].set_title("FFT Magnitude Spectrum")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI demo (generates a synthetic square wave if no real hardware present)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run FFT on a synthetic GPIO square-wave buffer (demo)."
    )
    parser.add_argument("--freq", type=float, default=5.0,
                        help="Synthetic signal frequency in Hz (default: 5)")
    parser.add_argument("--sample-rate", type=float, default=200.0,
                        help="Synthetic sample rate in Hz (default: 200)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Duration of synthetic signal in seconds (default: 2)")
    parser.add_argument("--window", type=str, default="hann",
                        choices=["hann", "hamming", "blackman", "none"])
    parser.add_argument("--plot", action="store_true",
                        help="Show matplotlib plot of results")
    args = parser.parse_args()

    # Build a synthetic square wave and push into a heap buffer
    print(f"[demo] Generating {args.freq} Hz square wave "
          f"at {args.sample_rate} Hz for {args.duration} s ...")

    buf = GPIOHeapBuffer(max_entries=int(args.sample_rate * args.duration) + 10)
    t = 0.0
    dt = 1.0 / args.sample_rate
    while t <= args.duration:
        # Square wave: HIGH for first half-period, LOW for second
        state = int((t * args.freq) % 1.0 < 0.5)
        # Add a tiny amount of jitter to simulate real hardware
        jitter = float(np.random.uniform(-dt * 0.05, dt * 0.05))
        buf.push(t + jitter, state)
        t += dt

    result = analyse(buf, window=args.window)
    result.print_summary()

    if args.plot:
        plot(result)
