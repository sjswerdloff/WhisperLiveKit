"""Regression test: concurrent MLX encoder calls must not abort the process.

TranscriptionEngine is a singleton and hands the SAME mlx_encoder to every
session, while transcription runs on asyncio's default thread pool. Two live
sessions therefore put two threads into the encoder at once. MLX allocates one
Metal command encoder per stream, and nothing on this path requests a stream
other than the default, so an overlapping pair aborts the whole process:

    tryCoalescingPreviousComputeCommandEncoderWithConfig:1094:
    failed assertion `A command encoder is already encoding to this command buffer'

That is abort(), so the failure CANNOT be observed as an exception from inside
the process. Each trial therefore runs as a SUBPROCESS and the observable is its
exit status: negative (killed by SIGABRT/SIGSEGV) versus 0.

The positive control is the load-bearing half. `test_unserialized_still_aborts`
runs the same code with the lock disabled and REQUIRES a crash. Without it, a
green run of the serialized case is indistinguishable from a test that stopped
exercising the race at all.
"""

import os
import subprocess
import sys
import threading

import pytest

MODEL = os.environ.get("WLK_TEST_MLX_MODEL", "mlx-community/whisper-tiny-mlx")
THREADS = 2
ITERS = 20
N_FRAMES = 3000

mlx_whisper = pytest.importorskip("mlx_whisper", reason="MLX backend not installed (non-Apple platform)")


def _run_trial(model_lock: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, WHISPERLIVEKIT_MODEL_LOCK=model_lock)
    return subprocess.run(
        [sys.executable, __file__, "--worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_unserialized_still_aborts():
    """POSITIVE CONTROL. With the lock off, overlapping encodes must kill the process.

    If this ever passes, the test below has stopped proving anything and the
    suite is green for the wrong reason -- fix this before trusting the other one.
    """
    proc = _run_trial("0")
    assert proc.returncode < 0, (
        "positive control did not fire: unserialized concurrent encodes were expected "
        f"to abort, got returncode={proc.returncode}. The serialization test below is "
        f"no longer evidence of anything.\nstdout={proc.stdout}\nstderr={proc.stderr[-500:]}"
    )


def test_serialized_survives():
    proc = _run_trial("1")
    assert proc.returncode == 0, (
        f"serialized concurrent encodes crashed: returncode={proc.returncode} "
        f"(negative = killed by signal)\nstdout={proc.stdout}\nstderr={proc.stderr[-500:]}"
    )


def _worker():
    """Subprocess body: THREADS threads, barrier-aligned, sharing one encoder."""
    import mlx.core as mx

    from whisperlivekit.simul_whisper.mlx_encoder import load_mlx_encoder
    from whisperlivekit.simul_whisper.simul_whisper import mlx_encode_serialized

    enc = load_mlx_encoder(path_or_hf_repo=MODEL)
    # One mel per thread, so a shared input buffer cannot be the explanation.
    mels = [mx.random.normal((N_FRAMES, enc.dims.n_mels)) for _ in range(THREADS)]
    mx.eval(mels)

    barrier = threading.Barrier(THREADS)
    errors = []

    def worker(idx):
        mel = mels[idx][None]
        try:
            for _ in range(ITERS):
                barrier.wait()  # force the overlap the production race only hits by luck
                mlx_encode_serialized(enc, mel)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            barrier.abort()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        print("PYERROR " + " | ".join(errors[:3]), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.exit(_worker())
    raise SystemExit("run via pytest; --worker is the subprocess entry point")
