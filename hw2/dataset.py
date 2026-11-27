import csv
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np

REVISION = "10f231ee2d43d6e38c777cbbed299f9ecd1c2d07"
SOURCE = (
    f"https://raw.githubusercontent.com/dirkhovy/MACE/{REVISION}/data/test"
)
CHECKSUMS = {
    "rte.csv": (
        "2b8767c19a5e0e79cdfb31ffbf61d7dd3dedb0eeaa71eb8230dba6ac80efdc2c"
    ),
    "rte.key": (
        "2d158af98a589ae2e0d746b1a5aed1f20f1683090f844bd41ff5392333f1e45d"
    ),
}
VOTE = {"": -1, "0": 0, "1": 1}


def read_source(directory: Path, name: str) -> str:
    path = directory / name
    expected = CHECKSUMS[name]
    if path.is_file():
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() == expected:
            return content.decode("ascii")

    with urlopen(f"{SOURCE}/{name}", timeout=30) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Unexpected SHA-256 for {name}")

    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory) as staging:
        temporary = Path(staging) / name
        temporary.write_bytes(content)
        temporary.replace(path)
    return content.decode("ascii")


def load_rte(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the task-by-worker vote matrix and aligned expert labels."""
    rows = csv.reader(read_source(directory, "rte.csv").splitlines())
    votes = np.array(
        [[VOTE[cell] for cell in row] for row in rows], dtype="i1"
    )
    labels = read_source(directory, "rte.key").splitlines()
    gold = np.array([int(label) for label in labels], dtype="i1")
    if len(gold) != len(votes) or not np.isin(gold, (0, 1)).all():
        raise ValueError("Expected one binary expert label per task")
    return votes, gold
