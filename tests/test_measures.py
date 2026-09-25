"""Measure registry, plugins, and the shape extra's interpretation bands."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import lqtpy
from lqtpy import registry
from lqtpy.manifest import Manifest
from lqtpy.measures import shape

import synthetic as S


def test_core_and_extra_measures_registered():
    names = {m.name: m.tier for m in registry.list_measures()}
    assert names["morphometry"] == "core"
    assert names["shape"] == "extra"


@pytest.mark.parametrize("mask,level", [
    (lambda: S.sphere([-40, 0, 10], 12), "compact"),
    (lambda: S.ellipsoid([-40, 0, 10], (6, 12, 6)), "elongated"),
    (lambda: S.ellipsoid([-40, 0, 10], (6, 20, 6)), "elongated-marked"),
    (lambda: S.ellipsoid([-40, 0, 10], (16, 16, 3)), "flat"),
    (lambda: S.blobs([([-40 + 11 * np.cos(a), 11 * np.sin(a), 10 + 3 * (-1) ** i], 7)
                      for i, a in enumerate(np.linspace(0, 2 * np.pi, 6)[:-1])]), "lobulated"),
])
def test_shape_bands(mask, level):
    v = shape.compute(lqtpy.prepare(S.img(mask())))
    assert level in [d.level for d in shape.describe_shape(v)]


def test_shape_small_lesion_not_characterised():
    v = shape.compute(lqtpy.prepare(S.img(S.sphere([-12, -18, 8], 2))))
    descs = shape.describe_shape(v)
    assert descs[0].level == "too-small"
    assert shape.sentence(descs) == "The lesion is too small to characterise its shape."


def test_shape_solidity_is_size_independent_for_spheres():
    for r in (5, 10, 20):
        v = shape.compute(lqtpy.prepare(S.img(S.sphere([-40, 0, 10], r))))
        assert v["solidity"] > 0.97 and v["surface_regularity"] > 0.9


def test_run_measures_records_manifest():
    m = Manifest("single")
    out = registry.run_measures(lqtpy.prepare(S.img(S.sphere([-40, 0, 10], 8))), manifest=m)
    assert set(out) >= {"morphometry", "shape"}
    d = m.to_dict()
    assert d["scope"] == "core + extras"
    assert {x["id"]: x["status"] for x in d["modules"]}["measure:shape"] == "ran"


def test_plugin_via_entry_point_and_failure_isolation(monkeypatch):
    """A plugin measure is discovered through entry points; a broken plugin
    and a measure with a missing dependency are reported, not raised."""
    mod = types.ModuleType("fake_lqtpy_plugin")
    mod.MEASURES = [registry.MeasureSpec(
        name="voxel_count_x2", label="Twice the voxels", tier="extra",
        compute=lambda p: {"x2": 2 * p.n_voxels})]
    mod.NEEDS = registry.MeasureSpec(name="needs_missing", label="Needs a missing lib",
                                     compute=lambda p: {}, requires=("no_such_module_xyz",))
    sys.modules["fake_lqtpy_plugin"] = mod

    class EP:
        def __init__(self, name, target):
            self.name, self._t = name, target

        def load(self):
            if self._t == "boom":
                raise RuntimeError("plugin exploded")
            return self._t

    eps = {"lqtpy.measures": [EP("fake", mod.MEASURES), EP("needs", mod.NEEDS), EP("broken", "boom")],
           "lqtpy.atlases": []}
    monkeypatch.setattr(registry, "_entry_points", lambda g: eps.get(g, []))
    registry.discover(force=True)
    try:
        p = lqtpy.prepare(S.img(S.sphere([-40, 0, 10], 5)))
        m = Manifest("single")
        out = registry.run_measures(p, manifest=m)
        assert out["voxel_count_x2"]["values"]["x2"] == 2 * p.n_voxels
        status = {x["id"]: x for x in m.to_dict()["modules"]}
        assert status["measure:needs_missing"]["status"] == "unavailable"
        assert "no_such_module_xyz" in status["measure:needs_missing"]["reason"]
        un = {u["name"]: u for u in registry.unavailable_measures()}
        assert "plugin exploded" in un["broken"]["reason"]
    finally:
        for n in ("voxel_count_x2", "needs_missing"):
            registry._MEASURES.pop(n, None)
        registry._UNAVAILABLE.clear()
        sys.modules.pop("fake_lqtpy_plugin", None)


def test_plugin_atlas_is_lazy(monkeypatch):
    calls = []
    shp = (40, 40, 40)
    aff = np.diag([2.0, 2, 2, 1])
    aff[:3, 3] = -40
    lab = np.zeros(shp, np.int16)
    lab[:20] = 1

    def load():
        calls.append(1)
        return (lab, aff), {1: {"RegionName": "PluginRegion", "Group": "G", "Lobe": "Frontal"}}

    spec = registry.AtlasSpec(key="plugatlas", name="Plugin atlas", load=load,
                              license="CC0", citation="none")

    class EP:
        name = "plug"

        def load(self):
            return spec

    monkeypatch.setattr(registry, "_entry_points",
                        lambda g: [EP()] if g == "lqtpy.atlases" else [])
    registry.discover(force=True)
    try:
        assert "plugatlas" in lqtpy.list_atlases()
        assert calls == []                       # listing doesn't load
        recs = lqtpy.atlas_overlap(S.img(S.sphere([-10, 0, 0], 6)), "plugatlas")
        assert calls == [1] and recs[0]["RegionName"] == "PluginRegion"
        assert lqtpy.lobe_overlap(S.img(S.sphere([-10, 0, 0], 6)), "plugatlas")[0]["Lobe"] == "Frontal"
        assert lqtpy.atlases.atlas_catalog()["plugatlas"]["source"] == "plugin:plug"
    finally:
        lqtpy.atlases._CUSTOM.pop("plugatlas", None)
        lqtpy.atlases._CACHE.pop("plugatlas", None)
