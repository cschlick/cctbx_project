"""
Regression test for per-atom density weighting in real-space refinement
(the ``weights`` argument threaded C++ compute -> real_space_refinement_simple
-> individual_sites).

Scenario (self-contained): half the model sits in GOOD density (map agrees with
the truth), half in POOR density (map peaks displaced -> misleading, a stand-in
for low local resolution). Refine the same shaken model against the same map
with (A) uniform weights and (B) quality weights (high in good region, low in
poor region), and check that weighting protects the poor region while matching
the good region. Also check that all-ones weights reproduce the unweighted path.
"""
from __future__ import absolute_import, division, print_function
import iotbx.pdb, mmtbx.model
from libtbx.utils import null_out
from libtbx.test_utils import approx_equal
from scitbx.array_family import flex
from scitbx import matrix
from mmtbx.refinement.real_space import individual_sites
from mmtbx.refinement.real_space.tst_rsr_peratom import pdb_str

def get_model():
  m = mmtbx.model.manager(model_input=iotbx.pdb.input(source_info=None, lines=pdb_str),
                          log=null_out())
  m.process(make_restraints=True)
  return m

def map_from_sites(xrs, sites, d_min=2.0):
  x = xrs.deep_copy_scatterers(); x.set_sites_cart(sites)
  fft = x.structure_factors(d_min=d_min).f_calc().fft_map(resolution_factor=0.25)
  fft.apply_sigma_scaling()
  return fft.real_map_unpadded()

def refine(xrs_answer, sites_start, grm_geo, target_map, weights):
  xrs = xrs_answer.deep_copy_scatterers()
  xrs.set_sites_cart(sites_start)
  ro = individual_sites.simple(
    target_map                  = target_map,
    selection                   = flex.bool(xrs.scatterers().size(), True),
    geometry_restraints_manager = grm_geo,
    gradients_method            = "tricubic",
    weights                     = weights,
    max_iterations              = 100)
  ro.refine(weight=20.0, xray_structure=xrs)
  return ro.sites_cart()

def exercise():
  flex.set_random_seed(0)
  m = get_model()
  xrs = m.get_xray_structure()
  grm_geo = m.get_restraints_manager().geometry
  n = xrs.scatterers().size()
  sites_true = xrs.sites_cart().deep_copy()

  poor = flex.bool([i >= n//2 for i in range(n)])
  good = ~poor

  # mixed-quality map: poor-region peaks displaced (misleading density)
  disp = flex.vec3_double(n, (0,0,0))
  for i in range(n):
    if poor[i]:
      v = matrix.col((flex.random_double()-0.5, flex.random_double()-0.5,
                      flex.random_double()-0.5))
      disp[i] = (v.normalize()*0.7).elems
  target_map = map_from_sites(xrs, sites_true + disp)

  xrs_s = xrs.deep_copy_scatterers(); xrs_s.shake_sites_in_place(mean_distance=0.3)
  sites_start = xrs_s.sites_cart()

  def dist(sites, sel):
    d = (sites_true - sites).select(sel)
    return flex.mean(flex.sqrt(d.dot()))

  # (A) uniform weighting (weights=None -> unweighted C++ path)
  r_uniform = refine(xrs, sites_start, grm_geo, target_map, None)
  # (B) quality weighting: good=1.0, poor=0.1, normalized to mean 1
  wq = flex.double([1.0 if good[i] else 0.1 for i in range(n)])
  wq = wq / flex.mean(wq)
  r_weighted = refine(xrs, sites_start, grm_geo, target_map, wq)
  # (C) all-ones weights must reproduce the unweighted path
  r_ones = refine(xrs, sites_start, grm_geo, target_map, flex.double(n, 1.0))

  print("uniform :  good=%.3f  POOR=%.3f" % (dist(r_uniform, good), dist(r_uniform, poor)))
  print("weighted:  good=%.3f  POOR=%.3f" % (dist(r_weighted, good), dist(r_weighted, poor)))

  # weighting protects the poor region ...
  assert dist(r_weighted, poor) < dist(r_uniform, poor)
  # ... without hurting the good region
  assert dist(r_weighted, good) < dist(r_uniform, good) + 0.02
  # all-ones weights == unweighted path (byte-identical C++ target/gradients)
  assert approx_equal(dist(r_ones, poor), dist(r_uniform, poor), eps=1.e-6)

if __name__ == "__main__":
  exercise()
  print("OK")
