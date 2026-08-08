"""Report what this machine will train on.

Run on each machine before a training session. The project targets two very
different hosts — a CPU-only laptop and a desktop with a GPU — and the throughput
numbers in the plan assume you know which one you are on.

    python scripts/detect_device.py
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"python   {sys.version.split()[0]}  ({platform.machine()})")
    print(f"system   {platform.system()} {platform.release()}")
    try:
        import os

        print(f"cpus     {os.cpu_count()} logical")
    except Exception:
        pass

    try:
        import torch
    except ImportError:
        print("\ntorch    not installed")
        print("         pip install -r requirements-train.txt, then reinstall torch")
        print("         with the index-url matching this machine — see that file.")
        return 1

    print(f"torch    {torch.__version__}")

    backends: list[tuple[str, str]] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            backends.append(
                (
                    f"cuda:{i}",
                    f"{props.name}, {props.total_memory / 1024**3:.1f} GiB, "
                    f"sm_{props.major}{props.minor}",
                )
            )
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        backends.append(("mps", "Apple Metal"))
    try:
        import torch_directml  # type: ignore

        backends.append((f"privateuseone:0", torch_directml.device_name(0)))
    except ImportError:
        pass

    if backends:
        print("\naccelerators:")
        for name, detail in backends:
            print(f"  {name:18} {detail}")
        chosen = backends[0][0]
    else:
        print("\naccelerators: none detected — training will run on CPU")
        print(f"              torch threads: {torch.get_num_threads()}")
        chosen = "cpu"

    print(f"\nwould train on: {chosen}")
    if chosen == "cpu":
        print("expect roughly 15-30k agent-decisions/s; a full run is days, not hours.")
    else:
        print("expect a large speedup over the CPU baseline on the network-bound stages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
