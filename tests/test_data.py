import hashlib
import os
import zipfile

import pytest

from craniofea import data


@pytest.fixture()
def fake_archive(tmp_path, monkeypatch):
    """A small zip laid out like an Open-Full-Jaw archive, registered as patient 99."""
    z = tmp_path / "src" / "Patient_99.zip"
    z.parent.mkdir()
    with zipfile.ZipFile(z, "w") as zf:
        for m in data.DEFAULT_MEMBERS:
            zf.writestr(f"Patient_99/{m}", f"content of {m}")
        zf.writestr("Patient_99/input/maxilla/bone/maxilla.stl", "not extracted")
    digest = hashlib.sha256(z.read_bytes()).hexdigest()
    monkeypatch.setitem(data.ARCHIVES, 99, (digest, z.stat().st_size))
    return z


def test_sha256_and_pointer_detection(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert data.sha256sum(p) == hashlib.sha256(b"abc").hexdigest()
    ptr = tmp_path / "Patient_3.zip"
    ptr.write_text("version https://git-lfs.github.com/spec/v1\n"
                   f"oid sha256:{data.ARCHIVES[3][0]}\nsize {data.ARCHIVES[3][1]}\n")
    assert data.is_lfs_pointer(ptr) and not data.is_lfs_pointer(p)
    with pytest.raises(ValueError, match="LFS pointer"):
        data.verify_archive(ptr, 3)


def test_dataset_url_is_pinned_lfs_media():
    url = data.dataset_url(3)
    assert url.startswith("https://media.githubusercontent.com/media/diku-dk/Open-Full-Jaw/")
    assert data.COMMIT in url and url.endswith("/dataset/Patient_3.zip")
    assert set(data.ARCHIVES) == set(range(1, 18))


def test_fetch_from_local_extracts_only_needed(fake_archive, tmp_path, capsys):
    dest = tmp_path / "ofj"
    out = data.fetch_open_full_jaw(99, dest, source=fake_archive)
    assert "CC BY-NC-SA 4.0" in capsys.readouterr().err
    for m in data.DEFAULT_MEMBERS:
        assert (out / "Patient_99" / m).read_text() == f"content of {m}"
    assert not (out / "Patient_99/input/maxilla/bone/maxilla.stl").exists()
    # A second call reuses the verified archive.
    assert data.fetch_open_full_jaw(99, dest, source=tmp_path / "missing.zip") == out


def test_fetch_rejects_corrupt_archive(fake_archive, tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(fake_archive.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="sha256"):
        data.fetch_open_full_jaw(99, tmp_path / "ofj", source=bad)


def test_cli_reports_errors(tmp_path, capsys):
    rc = data.main(["--patient", "3", "--dest", str(tmp_path), "--from-local",
                    str(tmp_path / "nope.zip")])
    assert rc == 2
    assert "error" in capsys.readouterr().err


@pytest.mark.skipif(not os.environ.get("CRANIOFEA_NETWORK"), reason="set CRANIOFEA_NETWORK=1")
def test_download_url_resolves():
    status, size = data.check_url(3)
    assert status == 200 and size == data.ARCHIVES[3][1]
