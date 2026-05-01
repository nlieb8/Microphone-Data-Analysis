"""
gpio_csv_writer.py
------------------
Companion module to gpio_logger.py.

Provides two ways to write GPIO samples to CSV:

1. CSVWriter          -- streaming writer, appends one row per sample in real time.
2. snapshot_to_csv()  -- one-shot export of a GPIOHeapBuffer snapshot to a file.

CSV format (both modes):
    timestamp,state
    1735000000.123456,HIGH
    1735000000.223456,LOW
    ...

Usage (streaming):
    from gpio_csv_writer import CSVWriter
    writer = CSVWriter("gpio_log.csv")
    writer.write(timestamp, state)   # call this in your sample loop
    writer.close()                   # flush + close when done

Usage (snapshot export):
    from gpio_csv_writer import snapshot_to_csv
    from gpio_logger import GPIOLogger

    logger = GPIOLogger(pin=17, freq=10, max_entries=200)
    logger.start()                             # blocks until Ctrl+C
    snapshot_to_csv(logger.get_snapshot(), "gpio_snapshot.csv")
"""

import csv
import io
import os
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADER = ["timestamp", "state"]

def _state_label(state: int) -> str:
    return "HIGH" if state else "LOW"


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------

class CSVWriter:
    """
    Opens a CSV file and appends one row per sample as they arrive.

    Args:
        filepath    -- path to output CSV (created or appended to)
        append      -- if True and file exists, append without re-writing header
        flush_every -- fsync to disk every N rows (0 = never force-flush)
    """

    def __init__(self, filepath: str | os.PathLike, append: bool = False, flush_every: int = 0):
        self.filepath = Path(filepath)
        self.flush_every = flush_every
        self._row_count = 0

        file_exists = self.filepath.exists() and self.filepath.stat().st_size > 0
        mode = "a" if (append and file_exists) else "w"

        self._fh = self.filepath.open(mode, newline="", buffering=1)  # line-buffered
        self._writer = csv.writer(self._fh)

        if not (append and file_exists):
            self._writer.writerow(_HEADER)

        print(f"[csv_writer] Logging to '{self.filepath}' (mode={mode})")

    def write(self, timestamp: float, state: int) -> None:
        """Append a single (timestamp, state) sample."""
        self._writer.writerow([f"{timestamp:.6f}", _state_label(state)])
        self._row_count += 1
        if self.flush_every and self._row_count % self.flush_every == 0:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def write_many(self, samples: list[tuple[float, int]]) -> None:
        """Append a batch of (timestamp, state) samples."""
        self._writer.writerows(
            [f"{ts:.6f}", _state_label(s)] for ts, s in samples
        )
        self._row_count += len(samples)

    def close(self) -> None:
        """Flush and close the file."""
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()
            print(f"[csv_writer] Closed '{self.filepath}' ({self._row_count} rows written)")

    # Context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        return (
            f"CSVWriter(path='{self.filepath}', "
            f"rows={self._row_count}, "
            f"closed={self._fh.closed})"
        )


# ---------------------------------------------------------------------------
# Snapshot export
# ---------------------------------------------------------------------------

def snapshot_to_csv(
    samples: list[tuple[float, int]],
    filepath: str | os.PathLike,
    overwrite: bool = True,
) -> Path:
    """
    Write a sorted list of (timestamp, state) samples to a CSV file in one shot.

    Args:
        samples   -- list of (timestamp, state) tuples, e.g. from logger.get_snapshot()
        filepath  -- destination path
        overwrite -- if False and file exists, raise FileExistsError

    Returns:
        Path to the written file.
    """
    out = Path(filepath)
    if not overwrite and out.exists():
        raise FileExistsError(f"'{out}' already exists and overwrite=False")

    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_HEADER)
        writer.writerows([f"{ts:.6f}", _state_label(s)] for ts, s in samples)

    print(f"[csv_writer] Snapshot written to '{out}' ({len(samples)} rows)")
    return out


# ---------------------------------------------------------------------------
# Integration helper: GPIOLogger subclass with built-in CSV streaming
# ---------------------------------------------------------------------------

def make_logging_logger(
    pin: int,
    freq: float,
    max_entries: int,
    csv_path: str | os.PathLike,
    append: bool = False,
    flush_every: int = 0,
    verbose: bool = True,
):
    """
    Convenience factory: returns a GPIOLogger whose sample loop
    also writes every reading to a CSV in real time.

    Example:
        logger = make_logging_logger(pin=17, freq=10, max_entries=200,
                                     csv_path="run1.csv")
        logger.start()   # blocks; Ctrl+C stops sampling and closes CSV
    """
    # Import here to avoid circular deps if files are in the same package
    from gpio_logger import GPIOLogger

    writer = CSVWriter(csv_path, append=append, flush_every=flush_every)

    class _LoggingGPIOLogger(GPIOLogger):
        def start(self):
            import time, heapq, signal
            try:
                import RPi.GPIO as GPIO
            except ImportError:
                raise SystemExit("RPi.GPIO not available.")

            self._running = True
            print(
                f"[gpio_logger] Sampling GPIO {self.pin} at {self.freq} Hz"
                f" | buffer: {self.buffer.max_entries} | CSV: '{csv_path}'\n"
                f"Press Ctrl+C to stop.\n"
            )

            next_tick = time.monotonic()
            try:
                while self._running:
                    now = time.monotonic()
                    state = GPIO.input(self.pin)
                    self.buffer.push(now, state)
                    writer.write(now, state)

                    if self.verbose:
                        print(
                            f"  t={now:.6f}  GPIO{self.pin}="
                            f"{'HIGH' if state else 'LOW '}"
                            f"  buffer_len={len(self.buffer)}"
                        )

                    next_tick += self.period
                    sleep_for = next_tick - time.monotonic()
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    else:
                        next_tick = time.monotonic()

            except KeyboardInterrupt:
                pass
            finally:
                self.stop()
                writer.close()

    return _LoggingGPIOLogger(
        pin=pin, freq=freq, max_entries=max_entries, verbose=verbose
    )


# ---------------------------------------------------------------------------
# CLI: export a snapshot CSV from a running logger (demo / manual trigger)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from gpio_logger import GPIOLogger

    parser = argparse.ArgumentParser(
        description="Run GPIO logger and export final snapshot to CSV."
    )
    parser.add_argument("--pin", type=int, default=17)
    parser.add_argument("--freq", type=float, default=10.0)
    parser.add_argument("--max-entries", type=int, default=100)
    parser.add_argument("--output", type=str, default="gpio_snapshot.csv")
    parser.add_argument("--stream", action="store_true",
                        help="Also stream each sample to CSV in real time")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.stream:
        logger = make_logging_logger(
            pin=args.pin,
            freq=args.freq,
            max_entries=args.max_entries,
            csv_path=args.output,
            verbose=not args.quiet,
        )
        logger.start()
    else:
        logger = GPIOLogger(
            pin=args.pin,
            freq=args.freq,
            max_entries=args.max_entries,
            verbose=not args.quiet,
        )
        logger.start()
        snapshot_to_csv(logger.get_snapshot(), args.output)
