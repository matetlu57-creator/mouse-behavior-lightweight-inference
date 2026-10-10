"""Recoverable local directory/file swaps, including destinations on other volumes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(path, roots):
    """Reject escaping paths and Windows reparse points before touching payloads."""
    path = Path(path).absolute()
    for node in [path, *path.parents]:
        if node.is_symlink() or (node.exists() and getattr(node.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError(f'linked/reparse paths are not supported: {node}')
    if not any(path.resolve().is_relative_to(Path(root).resolve()) for root in roots):
        raise ValueError(f'path outside configured roots: {path}')
    return path


def tree_hashes(path):
    path = Path(path)
    if path.is_file():
        return {'.': sha(path)}
    result = {}
    for item in sorted(path.rglob('*')):
        safe_path(item, [path])
        if item.is_file():
            result[item.relative_to(path).as_posix()] = sha(item)
    return result


class SwapTransaction:
    """Journal before renames; a prepared transaction is always rolled back."""

    def __init__(self, root, source_root):
        self.root = Path(root).resolve()
        self.source_root = Path(source_root).resolve()
        self.roots = [self.root, self.source_root]
        self.journal = self.root / 'active_transaction.json'

    def _entries(self, data):
        txid = data['transaction_id']
        if len(txid) != 32 or any(c not in '0123456789abcdef' for c in txid):
            raise ValueError('invalid transaction identifier')
        for entry in data['entries']:
            target = safe_path(entry['target'], self.roots)
            if target.is_relative_to(self.source_root):
                if len(target.relative_to(self.source_root).parts) != 2:
                    raise ValueError('source transaction must target a sample directory')
            else:
                parts = target.relative_to(self.root).parts
                if parts not in [('decisions.json',), ('dataset_manifest.json',)] and not (
                    len(parts) == 3 and parts[0] == 'dataset'
                ):
                    raise ValueError('unregistered confirmed transaction target')
            rollback = safe_path(entry['rollback'], self.roots)
            if rollback != target.with_name(f'.{target.name}.review-rollback-{txid}'):
                raise ValueError('invalid rollback path')
            stage = safe_path(entry['stage'], self.roots) if entry['stage'] else None
            if stage and (stage.parent != target.parent or not stage.name.startswith(f'.review-stage-{txid}-')):
                raise ValueError('stage must be on the destination volume beside its target')
            yield entry, target, stage, rollback

    def _remove(self, path, expected):
        safe_path(path, self.roots)
        # A prior cleanup may have removed some files before a Windows sharing
        # violation. Every remaining file must still be an unchanged owned file.
        remaining = tree_hashes(path)
        if any(name not in expected or digest != expected[name] for name, digest in remaining.items()):
            raise RuntimeError(f'transaction payload changed externally: {path}')
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    def recover(self):
        if not self.journal.exists():
            return
        data = json.loads(self.journal.read_text(encoding='utf-8'))
        entries = list(self._entries(data))  # Validate the entire journal first.
        if data['phase'] not in {'prepared', 'committed'}:
            raise ValueError('invalid transaction phase')
        if data['phase'] == 'prepared':
            for entry, target, stage, rollback in reversed(entries):
                if rollback.exists():
                    if tree_hashes(rollback) != entry['old_hashes']:
                        raise RuntimeError(f'rollback copy changed: {rollback}')
                    if target.exists():
                        self._remove(target, entry['new_hashes'])
                    os.replace(rollback, target)
                elif entry['old_exists']:
                    if not target.exists() or tree_hashes(target) != entry['old_hashes']:
                        raise RuntimeError(f'cannot recover original target: {target}')
                elif target.exists():
                    self._remove(target, entry['new_hashes'])
                if stage and stage.exists():
                    self._remove(stage, entry['new_hashes'])
        else:
            archive = self.root / 'transactions' / data['transaction_id'] / 'receipt.json'
            write_json(archive, data)
            for entry, target, stage, rollback in entries:
                if rollback.exists():
                    self._remove(rollback, entry['old_hashes'])
                if stage and stage.exists():
                    self._remove(stage, entry['new_hashes'])
        self.journal.unlink()

    def apply(self, txid, swaps):
        self.recover()
        entries = []
        for target, stage, expected_old, expected_new in swaps:
            target = safe_path(target, self.roots)
            if target.exists() != (expected_old is not None) or (
                target.exists() and tree_hashes(target) != expected_old
            ):
                raise RuntimeError(f'target changed while preparing confirmation: {target}')
            if stage and tree_hashes(stage) != expected_new:
                raise RuntimeError(f'staged sample changed while preparing confirmation: {stage}')
            rollback = target.with_name(f'.{target.name}.review-rollback-{txid}')
            if rollback.exists():
                raise FileExistsError(rollback)
            entries.append({'target': str(target), 'stage': str(stage) if stage else None,
                            'rollback': str(rollback), 'old_exists': expected_old is not None,
                            'old_hashes': expected_old or {},
                            'new_hashes': expected_new})
        data = {'transaction_id': txid, 'phase': 'prepared', 'entries': entries}
        list(self._entries(data))
        write_json(self.journal, data)
        try:
            for entry, target, stage, rollback in self._entries(data):
                if target.exists() != entry['old_exists']:
                    raise RuntimeError(f'target existence changed during save: {target}')
                if target.exists():
                    if tree_hashes(target) != entry['old_hashes']:
                        raise RuntimeError(f'target changed during save: {target}')
                    os.replace(target, rollback)
                if stage:
                    if tree_hashes(stage) != entry['new_hashes']:
                        raise RuntimeError(f'stage changed during save: {stage}')
                    os.replace(stage, target)
            for entry, target, stage, rollback in self._entries(data):
                if entry['stage']:
                    if not target.exists() or tree_hashes(target) != entry['new_hashes']:
                        raise RuntimeError(f'installed payload mismatch: {target}')
                elif target.exists():
                    raise RuntimeError(f'deleted target remains: {target}')
            data['phase'] = 'committed'
            write_json(self.journal, data)
        except BaseException:
            self.recover()  # On sharing violations leave journal in place and block later writes.
            raise
        self.recover()
