"""
Regression test for mmtbx.maps.local_cc_star (per-residue local map-model CC
with a half-map derived CC* ceiling).

Self-contained: builds a synthetic model + map + two independently-noised half
maps, so there are no file dependencies. Exercises the CC* formula, the
perfect-model ceiling behaviour, and -- the point of a *local* metric -- that a
local model perturbation depresses the local score where the error is.
"""
from __future__ import absolute_import, division, print_function

from scitbx.array_family import flex
from libtbx.test_utils import approx_equal
from iotbx.map_model_manager import map_model_manager
from mmtbx.maps import local_cc_star as lcs


def exercise_cc_star_formula():
  # CC* = sqrt(2r/(1+r)): perfect -> 1, zero -> 0, monotonic, ceiling >= r.
  assert approx_equal(lcs.cc_star_from_cc_half(1.0), 1.0)
  assert approx_equal(lcs.cc_star_from_cc_half(0.0), 0.0)
  assert lcs.cc_star_from_cc_half(None) is None
  assert lcs.cc_star_from_cc_half(-1.0) == 0.0
  prev = -1.0
  for r in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    cs = lcs.cc_star_from_cc_half(r)
    assert cs > r            # ceiling always above the half-half CC
    assert cs > prev         # monotonically increasing
    assert cs <= 1.0 + 1e-9
    prev = cs


def _synthetic_case():
  """Dummy model + generated map + two independently-noised half maps."""
  mmm = map_model_manager()
  mmm.generate_map()
  mm_full = mmm.map_manager()
  md = mm_full.map_data()
  sigma = md.sample_standard_deviation()

  def make_half(seed, frac=0.6):
    flex.set_random_seed(seed)
    n = flex.random_double(md.size())
    n.reshape(md.accessor())
    return mm_full.customized_copy(map_data=md + (n - 0.5) * 2.0 * frac * sigma)

  mmm.add_map_manager_by_id(make_half(1), "map_manager_1")
  mmm.add_map_manager_by_id(make_half(2), "map_manager_2")
  mmm.set_scattering_table("electron")
  return mmm


def _score(mmm, model, radius=3.0):
  mmm.generate_map(model=model, d_min=3.0, map_id="model_map")
  def md(i): return mmm.get_map_manager_by_id(i).map_data()
  return lcs.per_residue_local_cc_star(
    map_data_full=md("map_manager"), model_map_data=md("model_map"),
    unit_cell=mmm.map_manager().crystal_symmetry().unit_cell(), model=model,
    map_data_half1=md("map_manager_1"), map_data_half2=md("map_manager_2"),
    radius=radius)


def exercise_perfect_model():
  mmm = _synthetic_case()
  mmm.box_all_maps_around_model_and_shift_origin(box_cushion=6.0)
  r = _score(mmm, mmm.model())
  assert r.n_residues == mmm.model().get_hierarchy().overall_counts().n_residues
  # map was generated from the model -> high fit, and the ceiling sits above it.
  assert r.cc_mapmodel_mean > 0.9, r.cc_mapmodel_mean
  assert r.cc_star_mean >= r.cc_mapmodel_mean, (r.cc_star_mean, r.cc_mapmodel_mean)
  # a matching model should not be flagged as over-ceiling (overfitting).
  assert r.n_over_ceiling == 0, r.n_over_ceiling


def exercise_localization():
  # Perturb two residues; their local scores must drop far below the rest.
  mmm = _synthetic_case()
  model = mmm.model()
  targets = ("224", "225")
  sites = model.get_sites_cart()
  moved = []
  for rg in model.get_hierarchy().only_model().chains()[0].residue_groups():
    if rg.resseq.strip() in targets:
      for atom in rg.atoms():
        x, y, z = sites[atom.i_seq]
        sites[atom.i_seq] = (x + 1.5, y + 1.5, z)
        moved.append(atom.i_seq)
  assert len(moved) > 0
  model.set_sites_cart(sites)
  mmm.set_model(model, overwrite=True)
  mmm.box_all_maps_around_model_and_shift_origin(box_cushion=6.0)
  r = _score(mmm, mmm.model())

  bad = [x.cc_mapmodel for x in r.residues if x.resseq in targets]
  good = [x.cc_mapmodel for x in r.residues if x.resseq not in targets]
  assert len(bad) == 2
  # every perturbed residue scores well below every unperturbed one.
  assert max(bad) < min(good) - 0.2, (bad, good)
  # the ceiling (map quality) is unchanged by the model error.
  assert r.cc_star_mean > 0.9, r.cc_star_mean


def run():
  exercise_cc_star_formula()
  exercise_perfect_model()
  exercise_localization()


if __name__ == "__main__":
  run()
  print("OK")
