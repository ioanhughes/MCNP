from mesh_bins_helper import plan_mesh_from_origin

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
