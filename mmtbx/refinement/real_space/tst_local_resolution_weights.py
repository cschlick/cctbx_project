"""
End-to-end test of local-resolution weighting for real-space refinement.

It documents and reproduces the validation that motivated the feature: weights
derived *blindly* from local map resolution (windowed d99) both (a) correctly
identify a poor-resolution region and (b) protect that region during weighted
real-space refinement, relative to uniform weighting.

Scenario (self-contained, no external files)
--------------------------------------------
A small peptide is used to build a deliberately non-uniform target map:
  * the low-x half of space is rendered at high resolution (1.8 A) from the
    TRUE coordinates -> sharp, trustworthy density;
  * the high-x half is rendered at low resolution (3.5 A) from DISPLACED
    coordinates -> blurry AND misleading density (a stand-in for a poorly
    resolved, mobile region).
So the high-x atoms sit in density that is both low-resolution (detectable by
windowed d99) and wrong (harmful to fit).

We then, WITHOUT telling the code which atoms are poor:
  1. compute per-atom weights from the map alone via
     local_resolution_weights.local_resolution_weights, and check the poor
     (high-x) region gets lower weight than the good (low-x) region;
  2. shake the whole model and refine it against the map with uniform weights
     and with the blind weights, and check the blind weights leave the poor
     region closer to the truth.

The absolute improvement on this tiny synthetic case is modest (the 20 A cell
compresses the resolution contrast a 10 A window can see); the test asserts the
*direction*, which is the robust, reproducible claim.
"""
from __future__ import absolute_import, division, print_function
import math
import iotbx.pdb, mmtbx.model
from libtbx.utils import null_out, user_plus_sys_time
from scitbx.array_family import flex
from scitbx import matrix
from cctbx import maptbx
import cctbx.maptbx.real_space_refinement_simple as rsrs
import scitbx.lbfgs
from mmtbx.refinement.real_space import local_resolution_weights
from mmtbx.refinement.real_space.tst_rsr_peratom import pdb_str

def get_model():
  m = mmtbx.model.manager(model_input=iotbx.pdb.input(source_info=None, lines=pdb_str),
                          log=null_out())
  m.process(make_restraints=True)
  return m

def map_on(fc, cg):
  fm = fc.fft_map(crystal_gridding=cg); fm.apply_sigma_scaling()
  return fm.real_map_unpadded()

def refine(xrs, sites_start, grm, target_map, weights):
  x = xrs.deep_copy_scatterers(); x.set_sites_cart(sites_start)
  r = rsrs.lbfgs(
    sites_cart=x.sites_cart(), density_map=target_map,
    gradients_method="tricubic", unit_cell=x.unit_cell(),
    geometry_restraints_manager=grm, weights=weights,
    real_space_target_weight=20.0, real_space_gradients_delta=0.25,
    lbfgs_termination_params=scitbx.lbfgs.termination_parameters(max_iterations=100))
  return r.sites_cart

def exercise():
  flex.set_random_seed(0)
  m = get_model()
  xrs = m.get_xray_structure()
  grm = m.get_restraints_manager().geometry
  uc = xrs.unit_cell()
  n = xrs.scatterers().size()
  sites_true = xrs.sites_cart().deep_copy()
  sf = uc.fractionalize(sites_true)
  poor = flex.bool([sf[i][0] >= 0.5 for i in range(n)])   # high-x = poor region
  good = ~poor

  # displace poor-region atoms in the map-model (misleading density)
  disp = flex.vec3_double(n, (0,0,0))
  for i in range(n):
    if poor[i]:
      v = matrix.col((flex.random_double()-0.5, flex.random_double()-0.5,
                      flex.random_double()-0.5))
      disp[i] = (v.normalize()*0.7).elems
  xmm = xrs.deep_copy_scatterers(); xmm.set_sites_cart(sites_true + disp)

  cg = maptbx.crystal_gridding(unit_cell=uc,
    space_group_info=xrs.space_group_info(), d_min=1.8, resolution_factor=0.25)
  fc = xmm.structure_factors(d_min=1.8).f_calc()
  map_hi = map_on(fc, cg)
  map_lo = map_on(fc.resolution_filter(d_min=3.5), cg)
  # smooth spatial blend: high-res at low x, low-res at high x
  mask = map_hi.deep_copy(); md = mask.as_1d()
  nx, ny, nz = map_hi.all(); nyz = ny*nz
  for idx in range(md.size()):
    md[idx] = 1.0/(1.0 + math.exp(((idx//nyz)/nx - 0.5)/0.04))
  inv = mask.deep_copy(); inv *= -1.0; inv += 1.0
  target_map = mask*map_hi + inv*map_lo

  # (1) blind weights from local resolution
  weights = local_resolution_weights.local_resolution_weights(
    map_data=target_map, unit_cell=uc, sites_cart=sites_true, window_edge=10.0)
  w_good = flex.mean(weights.select(good))
  w_poor = flex.mean(weights.select(poor))
  print("blind local-resolution weights: good=%.3f  poor=%.3f" % (w_good, w_poor))
  assert w_poor < w_good      # poor region correctly down-weighted, blindly

  # (2) weighted RSR protects the poor region
  xs = xrs.deep_copy_scatterers(); xs.shake_sites_in_place(mean_distance=0.3)
  sites_start = xs.sites_cart()
  def dpoor(s):
    d = (sites_true - s).select(poor); return flex.mean(flex.sqrt(d.dot()))
  d_uniform  = dpoor(refine(xrs, sites_start, grm, target_map, None))
  d_weighted = dpoor(refine(xrs, sites_start, grm, target_map, weights))
  print("poor-region dist to truth: uniform=%.3f  weighted=%.3f" % (
    d_uniform, d_weighted))
  assert d_weighted < d_uniform

if (__name__ == "__main__"):
  timer = user_plus_sys_time()
  exercise()
  print("OK  Time: %6.2f" % timer.elapsed())
