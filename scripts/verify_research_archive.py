"""Verify a research ZIP or extracted archive using only the Python standard library."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def verify(path):
    path = Path(path)
    archive = zipfile.ZipFile(path) if path.is_file() else None
    try:
        if archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate ZIP entry')
            actual = set(names)
            read = archive.read
        else:
            actual = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}
            read = lambda name: (path/name).read_bytes()
        manifest = json.loads(read('ARCHIVE_MANIFEST.json'))
        if actual != set(manifest['files']) | {'ARCHIVE_MANIFEST.json'}:
            raise ValueError('Archive payload file set differs from manifest')
        for name, expected in manifest['files'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive member path')
            if not archive and ((path/name).is_symlink() or not (path/name).resolve().is_relative_to(path.resolve())):
                raise ValueError('Archive member escapes root')
            payload = read(name)
            if len(payload) != expected['bytes'] or hashlib.sha256(payload).hexdigest() != expected['sha256']:
                raise ValueError('Archive integrity failure: '+name)
        return {'status': 'passed', 'files_verified': len(manifest['files']),
                'reviewed_plan_sha256': manifest['reviewed_plan_sha256'],
                'test_status': manifest['test_status']}
    finally:
        if archive:
            archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    print(json.dumps(verify(parser.parse_args().path), indent=2))
