"""Per-episode transactions for offline processors working on a dataset copy."""

import os
from pathlib import Path
import shutil
import tempfile

import h5py


def _file_state(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _require_self_contained(h5_file):
    """Reject indirection that could let a temporary copy write another file."""
    def check_group(group):
        for name in group:
            if isinstance(group.get(name, getlink=True), h5py.ExternalLink):
                raise ValueError(f'External HDF5 link is unsupported: {group.name}/{name}')

    def check_object(name, obj):
        if isinstance(obj, h5py.Group):
            check_group(obj)
        elif isinstance(obj, h5py.Dataset) and (obj.external or obj.is_virtual):
            raise ValueError(f'Externally backed HDF5 dataset is unsupported: {name}')

    check_group(h5_file)
    h5_file.visititems(check_object)


def atomic_process_hdf5(path, transform):
    """Transform a sibling copy, publishing it only after a successful close.

    A failed transform, flush, or rename leaves the original episode intact.
    Run on an offline working copy: this is not a concurrent-writer protocol or
    a guarantee against power loss. One extra episode's disk space is required.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink episode: {path}")
    before = _file_state(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        shutil.copy2(path, temporary)
        with h5py.File(temporary, "r+") as h5_file:
            _require_self_contained(h5_file)
            transform(h5_file)
            h5_file.flush()
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        if path.is_symlink() or _file_state(path) != before:
            raise RuntimeError(f"Episode changed during processing; refusing replacement: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
