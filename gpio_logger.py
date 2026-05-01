"""
gpio_logger.py
--------------
Samples a single GPIO pin at a fixed clock frequency and maintains
a fixed-size min-heap buffer of the last N (timestamp, state) readings.
Oldest entries are evicted automatically when the buffer is full.

Usage:
    python gpio_logger.py --pin 17 --freq 10 --max-entries 100

Requirements:
    pip install RPi.GPIO
"""

import heapq
import time
import argparse
import signal
import sys

try:
    import RPi.GPIO as GPIO
except ImportError:
    raise SystemExit(
        "RPi.GPIO is not installed. Run: pip install RPi.GPIO\n"
        "This script must be run on a Raspberry Pi."
    )


# ---------------------------------------------------------------------------
# Min-heap buffer
# ---------------------------------------------------------------------------

class GPIOHeapBuffer:
    """
    A fixed-capacity min-heap of (timestamp, state) samples.

    The heap is ordered by timestamp (ascending), so the oldest entry
    is always at index 0 and can be evicted in O(log n).

    Public API:
        push(timestamp, state)  -- add a new sample, evict oldest if full
        peek_oldest()           -- return oldest (timestamp, state) without removing
        to_sorted_list()        -- return all entries oldest-first (does NOT mutate heap)
        __len__()
    """

    def __init__(self, max_entries: int):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries = max_entries
        self._heap: list[tuple[float, int]] = []   # (timestamp, state)

    def push(self, timestamp: float, state: int) -> None:
        entry = (timestamp, state)
        if len(self._heap) < self.max_entries:
            heapq.heappush(self._heap, entry)
        else:
            # Replace oldest only if this entry is newer (it always should be,
            # but guard against clock skew / monotonic weirdness).
            if timestamp > self._heap[0][0]:
                heapq.heapreplace(self._heap, entry)

    def peek_oldest(self) -> tuple[float, int] | None:
        return self._heap[0] if self._heap else None

    def to_sorted_list(self) -> list[tuple[float, int]]:
        """Return a sorted copy — O(n log n), use sparingly in hot loops."""
        return sorted(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __repr__(self) -> str:
        return (
            f"GPIOHeapBuffer(max={self.max_entries}, "
            f"used={len(self)}, "
            f"oldest={self.peek_oldest()})"
        )


# ---------------------------------------------------------------------------
# GPIO sampler
# ---------------------------------------------------------------------------

class GPIOLogger:
    """
    Samples a GPIO pin at `freq` Hz and stores readings in a GPIOHeapBuffer.

    Args:
        pin         -- BCM pin number to monitor
        freq        -- sampling frequency in Hz
        max_entries -- max entries kept in the heap buffer
        verbose     -- print each sample to stdout
    """

    def __init__(self, pin: int, freq: float, max_entries: int, verbose: bool = True):
        self.pin = pin
        self.freq = freq
        self.period = 1.0 / freq
        self.verbose = verbose
        self.buffer = GPIOHeapBuffer(max_entries)
        self._running = False

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN)

    def start(self) -> None:
        """Block and sample until stop() is called or KeyboardInterrupt."""
        self._running = True
        print(
            f"[gpio_logger] Sampling GPIO {self.pin} at {self.freq} Hz "
            f"| buffer size: {self.buffer.max_entries} entries\n"
            f"Press Ctrl+C to stop.\n"
        )

        next_tick = time.monotonic()

        try:
            while self._running:
                now = time.monotonic()
                state = GPIO.input(self.pin)
                self.buffer.push(now, state)

                if self.verbose:
                    print(f"  t={now:.6f}  GPIO{self.pin}={'HIGH' if state else 'LOW '}"
                          f"  buffer_len={len(self.buffer)}")

                # Drift-corrected sleep: aim for the next scheduled tick
                next_tick += self.period
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    # Fell behind — skip to the next future tick
                    next_tick = time.monotonic()

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        GPIO.cleanup()
        print(f"\n[gpio_logger] Stopped. {len(self.buffer)} entries in buffer.")

    def get_snapshot(self) -> list[tuple[float, int]]:
        """Return a sorted list of all buffered (timestamp, state) entries."""
        return self.buffer.to_sorted_list()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a Raspberry Pi GPIO pin and keep a rolling min-heap buffer."
    )
    parser.add_argument(
        "--pin", type=int, default=17,
        help="BCM GPIO pin number to monitor (default: 17)"
    )
    parser.add_argument(
        "--freq", type=float, default=10.0,
        help="Sampling frequency in Hz (default: 10)"
    )
    parser.add_argument(
        "--max-entries", type=int, default=100,
        help="Max entries to keep in the rolling buffer (default: 100)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-sample console output"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger = GPIOLogger(
        pin=args.pin,
        freq=args.freq,
        max_entries=args.max_entries,
        verbose=not args.quiet,
    )

    # Graceful SIGTERM handling (e.g. systemd)
    signal.signal(signal.SIGTERM, lambda *_: logger.stop())

    logger.start()

    # After stopping, you can inspect the buffer:
    snapshot = logger.get_snapshot()
    print(f"\n--- Final snapshot ({len(snapshot)} entries) ---")
    for ts, state in snapshot:
        print(f"  t={ts:.6f}  {'HIGH' if state else 'LOW'}")


if __name__ == "__main__":
    main()
