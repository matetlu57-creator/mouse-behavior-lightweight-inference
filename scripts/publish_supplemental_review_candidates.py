"""Publish prepared supplemental cards into an existing manual-review dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401
from mouse_behavior.evaluation.manual_review import ReviewStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--confirmed-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--candidate-document', type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.candidate_document.read_text(encoding='utf-8'))
    store = ReviewStore(args.package_root, args.confirmed_root, args.source_root)
    try:
        snapshot = store.publish_supplemental_candidates(
            document, expected_revision=store.snapshot()['revision'])
        print(json.dumps({
            'revision': snapshot['revision'],
            'supplemental_sha256': snapshot['supplemental_sha256'],
            'candidate_count': len(store.supplemental_cards),
            'review_ids': list(store.supplemental_cards),
        }, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == '__main__':
    main()
