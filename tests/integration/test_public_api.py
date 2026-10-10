from __future__ import annotations


def test_behavior_package_imports_without_legacy_full_pipeline():
    import mouse_behavior
    from mouse_behavior.core.pipeline import LightweightPipeline
    from mouse_behavior.identity.mtrack import IdentityMap

    assert mouse_behavior.__version__
    assert LightweightPipeline.__name__ == "LightweightPipeline"
    assert IdentityMap.__name__ == "IdentityMap"
