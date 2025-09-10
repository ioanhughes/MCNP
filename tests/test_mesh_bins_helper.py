import pytest

from mesh_bins_helper import plan_mesh_from_origin, plan_mesh_from_origin_counts

def test_plan_mesh_from_origin_uniform():
    origin = (0.0, 0.0, 0.0)
    result = plan_mesh_from_origin(origin, 6.0, 4.0, 3.0, delta=0.25)
    data = result["result"]
    assert data["iints"] == 24
    assert data["jints"] == 16
    assert data["kints"] == 12
    # ensure edges are computed
    edges = result["edges"]
    assert len(edges["x"]) == 25
    assert edges["x"][0] == origin[0]
    assert edges["x"][-1] == 6.0


def test_plan_mesh_from_origin_counts_uniform():
    origin = (0.0, 0.0, 0.0)
    result = plan_mesh_from_origin_counts(origin, 24, 16, 12, delta=0.25)
    ext = result["extents"]
    assert ext["xmax"] == pytest.approx(6.0)
    assert ext["ymax"] == pytest.approx(4.0)
    assert ext["zmax"] == pytest.approx(3.0)
    data = result["result"]
    assert data["iints"] == 24
    assert data["jints"] == 16
    assert data["kints"] == 12
