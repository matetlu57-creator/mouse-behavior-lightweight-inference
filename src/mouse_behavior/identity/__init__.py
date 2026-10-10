"""Identity adapters that map visual tracklets to experimental animals."""

from .mtrack import IdentityAssignment, IdentityMap, load_identity_map
from .labels import annotate_identity_labels, identity_label

__all__ = [
    "IdentityAssignment",
    "IdentityMap",
    "annotate_identity_labels",
    "identity_label",
    "load_identity_map",
]
