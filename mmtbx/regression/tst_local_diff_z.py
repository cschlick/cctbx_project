"""
Regression test for mmtbx.maps.local_diff_z (per-residue difference-density
Z-score / map-model accuracy).

Self-contained: a synthetic model + a map generated from it + two independently
noised half maps. The point of an *accuracy* metric is that displacing a residue
creates unexplained/excess density there, so diff_z must spike at the moved
residues while staying near noise level elsewhere.
"""
from __future__ import absolute_import, division, print_function

from scitbx.array_family import flex
from iotbx.map_model_manager import map_model_manager
from mmtbx.maps import local_diff_z as ldz
from mmtbx.maps.diff_map import compdiff


def _synthetic_case():
  """Dummy model + generated map + two independently-noised half maps."""
  mmm = map_model_manager()
  mmm.generate_map()
  mm_full = mmm.map_manager()
  md = mm_full.map_data()
  sigma = md.sample_standard_deviation()

  def make_half(seed, frac=0.5):
    flex.set_random_seed(seed)
    n = flex.random_double(md.size())
    n.reshape(md.accessor())
    return mm_full.customized_copy(map_data=md + (n - 0.5) * 2.0 * frac * sigma)

  mmm.add_map_manager_by_id(make_half(1), "map_manager_1")
  mmm.add_map_manager_by_id(make_half(2), "map_manager_2")
  return mmm


def _score(mmm, radius=2.0, d_min=3.0):
  model = mmm.model().remove_hydrogens()
  mmm.set_model(model, overwrite=True)
  model.setup_scattering_dictionaries(scattering_table="electron")
  residual = compdiff(
    map_data_obs=mmm.map_manager().map_data(),
    xrs=model.get_xray_structure(), d_min=d_min, vector_map=False,
    sigma_scale=False).map_result
  noise = (mmm.get_map_manager_by_id("map_manager_1").map_data()
           - mmm.get_map_manager_by_id("map_manager_2").map_data())
  return ldz.per_residue_diff_z(
    residual_map=residual, noise_map=noise,
    unit_cell=mmm.map_manager().crystal_symmetry().unit_cell(),
    model=model, radius=radius)


def exercise_perfect_model():
  # A model that matches the map should leave only noise-level residuals.
  mmm = _synthetic_case()
  mmm.box_all_maps_around_model_and_shift_origin(box_cushion=6.0)
  r = _score(mmm)
  assert r.n_residues > 0
  assert r.diff_z_mean_abs is not None
  # net residual is ~noise; mean |diff_z| should be modest (well below the 3-sigma
  # alarm) for a correct model.
  assert r.diff_z_mean_abs < 3.0, r.diff_z_mean_abs


def exercise_localization():
  # Displace two residues; their diff_z must spike far above the rest.
  mmm = _synthetic_case()
  model = mmm.model()
  targets = ("224", "225")
  sites = model.get_sites_cart()
  for rg in model.get_hierarchy().only_model().chains()[0].residue_groups():
    if rg.resseq.strip() in targets:
      for atom in rg.atoms():
        x, y, z = sites[atom.i_seq]
        sites[atom.i_seq] = (x + 1.2, y + 1.2, z)
  model.set_sites_cart(sites)
  mmm.set_model(model, overwrite=True)
  mmm.box_all_maps_around_model_and_shift_origin(box_cushion=6.0)
  r = _score(mmm)

  by = {x.resseq: abs(x.diff_z) for x in r.residues}
  bad = [by[t] for t in targets if t in by]
  good = [abs(x.diff_z) for x in r.residues if x.resseq not in targets]
  assert len(bad) == 2, bad
  # every displaced residue is a clear outlier: above the 3-sigma alarm and
  # well above the median unperturbed residue.
  good_sorted = sorted(good)
  median = good_sorted[len(good_sorted) // 2]
  assert min(bad) > 3.0, bad
  assert min(bad) > 3.0 * median, (bad, median)


def run():
  exercise_perfect_model()
  exercise_localization()


if __name__ == "__main__":
  run()
  print("OK")
