"""Download and verification of the Open-Full-Jaw dataset.

Open-Full-Jaw (Gholamalizadeh et al., "Open-Full-Jaw: An open-access dataset
and pipeline for finite element models of human jaw", *Comput. Methods
Programs Biomed.* 224:107009, 2022; https://github.com/diku-dk/Open-Full-Jaw)
distributes one zip archive per patient through git LFS. A plain ``raw.githubusercontent.com`` URL returns
only the 134-byte LFS *pointer*; the archive itself is served from
``media.githubusercontent.com``. URLs are pinned to the repository commit
below and every archive is verified against the sha256 recorded in its LFS
pointer (which is the sha256 of the archive).

The dataset is licensed CC BY-NC-SA 4.0. Meshes derived from it (everything
under ``data/``) must not be committed to this repository or redistributed
under a different license.

Command line::

    python -m craniofea.data --patient 3                       # download
    python -m craniofea.data --patient 3 --from-local PATH.zip # use a local copy
    python -m craniofea.data --patient 3 --check-url           # HEAD request only
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

REPOSITORY = "diku-dk/Open-Full-Jaw"
#: Commit of the dataset files (last change to dataset/Patient_*.zip, 2022-07-04).
COMMIT = "72b967df7943e97c84d5ac63fae85cf81ab76cb7"
URL_TEMPLATE = ("https://media.githubusercontent.com/media/{repo}/{ref}/dataset/"
                "Patient_{patient}.zip")

#: (sha256, size in bytes) of each patient archive, from the git-LFS pointers.
ARCHIVES: dict[int, tuple[str, int]] = {
    1: ("bf1553ff6c6b40aebe3115883be884b9b2a23e03c282ab657c71cd065ec94215", 52491493),
    2: ("f3604be58cbc871691fd64aa8afe299a705b4d31cb0608933f77e492346145ae", 86872297),
    3: ("efb5b6907fe50ff8322280b41dd2bd0c14dc40c2f0ad6fbbbb7f6b016061e9a4", 280633063),
    4: ("a507270b764ce0b08ceeb3377c372451ad9cf4b4b1475d7bd3d9031c9419bcb2", 258354477),
    5: ("1d4a3b5186c2f3b98278ab534bf423e79000f9985d80e4041af5f7fe2a1d3522", 260354375),
    6: ("ee560026075e94c56ff861f18aab76d35908b0c7eef3f1f98e71ab2fd705c197", 253662655),
    7: ("3ad3b7d44710c7fd032158666a99dc62c0b10946ac780923e2b585719442f151", 87540548),
    8: ("148b3badfe1303aca3c7ef4bb07aa33bf79c5c40cc6944a4ef54ce5cb20c4280", 236360850),
    9: ("2ed5ecf41503fae91c50aa0c583cbe9018059da4761c15bd171f53dfeb30eb6a", 130242444),
    10: ("f38d7d067cd28788e267935ade19248e9db5ee15da4d6817cccd3ed8b3c874bd", 115912622),
    11: ("404735c079f0f7232dda5ddc714018c02b925ddaed5d11d121e7f01c25280ab6", 237610643),
    12: ("ce112b9c97b6eecb9ce57ec3d5444dbd7280776a50ef0740e334f73294a60acd", 128445587),
    13: ("a139b9da3f8edd9a31c27575005193b83437b5207243ba8201b20eb9d89ddc7b", 140718244),
    14: ("a06b41bf013973ed580b1afb9c8698f47c33cf89e7dae8182da71f445e6fbc99", 117632788),
    15: ("11bea4d25fd1553d5b777c67d6017c45f083f9fbfd221fbf030b2f3deaca240f", 327772659),
    16: ("dded03f35852b3f47d65ca99209a0286cc00f7d62cb8da2508badcebeee0194f", 313111813),
    17: ("9c8bb25991d4b4c62661e0f01e2eeee769043d3354c2bb4fca9f574016cf2486", 194047366),
}

#: Archive members needed by the maxilla pipeline (relative to Patient_<n>/).
DEFAULT_MEMBERS = (
    "output/maxilla/volumetric_meshes/maxilla_volumetric_mesh.msh",
    "output/maxilla/parameters_log.json",
    "input/maxilla/teeth_axes_maxilla.json",
)

LICENSE_NOTICE = """\
Open-Full-Jaw dataset (Gholamalizadeh et al. 2022, Comput. Methods Programs Biomed.
224:107009, https://github.com/diku-dk/Open-Full-Jaw) is licensed under CC BY-NC-SA 4.0
(https://creativecommons.org/licenses/by-nc-sa/4.0/): attribution required,
non-commercial use only, derivatives under the same license. Meshes derived from it
(data/) must not be committed to this repository."""


def dataset_url(patient: int, ref: str = COMMIT) -> str:
    return URL_TEMPLATE.format(repo=REPOSITORY, ref=ref, patient=patient)


def sha256sum(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def is_lfs_pointer(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file() or p.stat().st_size > 1024:
        return False
    return p.read_bytes().startswith(b"version https://git-lfs.github.com/spec/")


def check_url(patient: int = 3, ref: str = COMMIT, timeout: float = 30.0) -> tuple[int, int]:
    """HEAD request for the archive URL. Returns (HTTP status, content length)."""
    req = urllib.request.Request(dataset_url(patient, ref), method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, int(r.headers.get("Content-Length", -1))


def verify_archive(path: str | Path, patient: int) -> None:
    """Raise ValueError unless `path` is the expected archive for `patient`."""
    path = Path(path)
    if is_lfs_pointer(path):
        raise ValueError(f"{path} is a git-LFS pointer, not the archive; install git-lfs "
                         f"or download {dataset_url(patient)}")
    expected, size = ARCHIVES[patient]
    if path.stat().st_size != size:
        raise ValueError(f"{path}: size {path.stat().st_size} != expected {size}")
    digest = sha256sum(path)
    if digest != expected:
        raise ValueError(f"{path}: sha256 {digest} != expected {expected}")


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh, length=1 << 20)
    tmp.replace(dest)


def extract_members(archive: str | Path, patient: int, members: tuple[str, ...],
                    out_dir: str | Path) -> list[Path]:
    """Extract only `members` (relative to Patient_<n>/) into out_dir/Patient_<n>/..."""
    out_dir = Path(out_dir)
    written = []
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        for m in members:
            name = f"Patient_{patient}/{m}"
            if name not in names:
                raise KeyError(f"{name} not in {archive}")
            target = out_dir / name
            if target.exists() and target.stat().st_size == z.getinfo(name).file_size:
                written.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            written.append(target)
    return written


def fetch_open_full_jaw(patient: int = 3, dest: str | Path = "data/open_full_jaw",
                        source: str | Path | None = None,
                        members: tuple[str, ...] = DEFAULT_MEMBERS,
                        link: bool = True) -> Path:
    """Obtain, verify and extract one Open-Full-Jaw patient archive.

    Parameters
    ----------
    patient : 1..17
    dest : directory receiving ``dataset/Patient_<n>.zip`` and
        ``extracted/Patient_<n>/...``
    source : None to download from the pinned GitHub LFS URL, an ``http(s)``
        URL, or a local archive path (``--from-local``), which is verified
        and then symlinked (``link=True``) or copied into `dest`
    members : archive members to extract (relative to ``Patient_<n>/``)

    Returns the extraction root ``dest/extracted``.
    """
    if patient not in ARCHIVES:
        raise ValueError(f"patient must be one of {sorted(ARCHIVES)}")
    print(LICENSE_NOTICE, file=sys.stderr)
    dest = Path(dest)
    archive = dest / "dataset" / f"Patient_{patient}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)

    ok = False
    if archive.exists() and not is_lfs_pointer(archive):
        try:
            verify_archive(archive, patient)
            ok = True
        except ValueError as err:
            log.warning("existing archive rejected: %s", err)
    if not ok:
        if archive.exists() or archive.is_symlink():
            archive.unlink()
        src = str(source) if source is not None else dataset_url(patient)
        if src.startswith(("http://", "https://")):
            _download(src, archive)
        else:
            local = Path(src).expanduser().resolve()
            verify_archive(local, patient)
            if link:
                try:
                    archive.symlink_to(local)
                except OSError:
                    shutil.copy2(local, archive)
            else:
                shutil.copy2(local, archive)
        verify_archive(archive, patient)
    log.info("verified %s (sha256 %s)", archive, ARCHIVES[patient][0][:12])
    out = dest / "extracted"
    for p in extract_members(archive, patient, members, out):
        log.info("extracted %s", p)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch an Open-Full-Jaw patient archive.")
    ap.add_argument("--patient", type=int, default=3)
    ap.add_argument("--dest", default="data/open_full_jaw")
    ap.add_argument("--from-local", metavar="PATH", help="use a local Patient_<n>.zip")
    ap.add_argument("--url", help="override the download URL")
    ap.add_argument("--copy", action="store_true", help="copy a local archive, do not link")
    ap.add_argument("--check-url", action="store_true", help="only send a HEAD request")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.check_url:
        status, size = check_url(args.patient)
        print(f"{dataset_url(args.patient)}: HTTP {status}, {size} bytes "
              f"(expected {ARCHIVES[args.patient][1]})")
        return 0 if status == 200 and size == ARCHIVES[args.patient][1] else 1
    try:
        fetch_open_full_jaw(args.patient, args.dest, args.from_local or args.url,
                            link=not args.copy)
    except (ValueError, KeyError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
