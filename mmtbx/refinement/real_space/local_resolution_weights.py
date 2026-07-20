"""
Per-atom weights from local map resolution, for weighted real-space refinement.

A local resolution is estimated at each atom by a systematic *windowed d99*:
cut a fixed-edge cube out of the map centered on the atom, apply a soft radial
taper, treat the cube as its own P1 cell, and run ``maptbx.d99`` on it. The soft
taper is essential -- a hard box edge injects high-frequency truncation
artifacts that make the windowed d99 read a much finer (wrong) resolution.

Each per-atom resolution is then mapped to a weight (better/smaller resolution
-> larger weight) as ``(median_resolution / resolution) ** power``, clamped and
normalized to mean 1 so the weights act *relatively* on the density term without
changing the overall density-vs-geometry balance.

The windowed-d99 estimate needs only a single map, but it is noisy at the
per-atom scale. When two half maps are available, prefer
``fsc_local_resolution_weights`` -- a per-atom local resolution from the
validated half-map FSC (``map_model_manager.local_resolution_map``), which is
the model-free gold standard. Both produce a weight array that plugs directly
into ``mmtbx.refinement.real_space.rsr_peratom.run(weights=...)`` (and the
``individual_sites`` / ``real_space_refinement_simple`` ``weights`` argument).
"""
from __future__ import absolute_import, division, print_function
import math
from cctbx import maptbx, crystal
from scitbx.array_family import flex


def _radial_cosine_taper(box_all, half_widths):
  """1-D flex.double (parallel to box.as_1d()) with a radial raised-cosine taper:
  1 at the box center, smoothly to 0 at radius min(half_widths)."""
  radius = float(min(half_widths))
  taper = flex.double(box_all[0]*box_all[1]*box_all[2])
  i = 0
  for ix in range(box_all[0]):
    for iy in range(box_all[1]):
      for iz in range(box_all[2]):
        r = math.sqrt((ix-half_widths[0])**2 +
                      (iy-half_widths[1])**2 +
                      (iz-half_widths[2])**2)
        taper[i] = 0.5*(1.0 + math.cos(math.pi*min(r/radius, 1.0)))
        i += 1
  return taper


def windowed_local_resolution(map_data, unit_cell, sites_cart, window_edge=10.0):
  """
  Per-atom local resolution (Angstrom) by windowed d99.

  Parameters
  ----------
  map_data : scitbx.array_family.flex.double
      Real-space map on the model's unit-cell grid.
  unit_cell : cctbx.uctbx.unit_cell
      Unit cell of the full map.
  sites_cart : scitbx.array_family.flex.vec3_double
      Atomic coordinates (Cartesian). Windows are centered on these.
  window_edge : float
      Edge length (Angstrom) of the cubic window. Roughly 2-4x the map
      resolution is sensible: too small and the box has too few reflections for
      d99; too large and the estimate loses spatial locality.

  Returns
  -------
  scitbx.array_family.flex.double
      Local resolution at each atom, parallel to sites_cart.
  """
  n = map_data.all()
  params = unit_cell.parameters()
  spacing = [params[i]/n[i] for i in range(3)]
  half_widths = [max(2, int(round((window_edge/2.0)/spacing[i]))) for i in range(3)]
  sites_frac = unit_cell.fractionalize(sites_cart)
  resolution = flex.double(sites_cart.size(), 0)
  taper = None
  for i_site in range(sites_cart.size()):
    sf = sites_frac[i_site]
    center = [int(round(sf[k]*n[k])) for k in range(3)]
    box = maptbx.copy(
      map_data,
      [center[k]-half_widths[k] for k in range(3)],
      [center[k]+half_widths[k] for k in range(3)])   # periodic sub-box (a copy)
    box_all = box.all()
    if taper is None:                                  # all boxes share a shape
      taper = _radial_cosine_taper(box_all, half_widths)
    box_1d = box.as_1d()
    box_1d *= taper                                    # soft taper, in place
    box_cs = crystal.symmetry(
      unit_cell=tuple([box_all[k]*spacing[k] for k in range(3)]) + (90.,90.,90.),
      space_group_symbol="P1")
    resolution[i_site] = maptbx.d99(
      map=box, crystal_symmetry=box_cs).result.d99
  return resolution


def weights_from_resolution(resolution, power=2.0,
                            weight_min=0.1, weight_max=10.0):
  """
  Map per-atom resolution to normalized per-atom weights (mean 1).

  weight = (median_resolution / resolution) ** power, clamped to
  [weight_min, weight_max], then divided by its mean. Better (smaller)
  resolution -> larger weight. ``power`` controls the sharpness of the mapping.
  """
  assert resolution.size() > 0
  assert flex.min(resolution) > 0
  d_ref = flex.median(resolution)
  w = flex.pow(d_ref / resolution, power)
  w.set_selected(w < weight_min, weight_min)
  w.set_selected(w > weight_max, weight_max)
  return w / flex.mean(w)


def local_resolution_weights(map_data, unit_cell, sites_cart,
                             window_edge=10.0, power=2.0,
                             weight_min=0.1, weight_max=10.0):
  """
  Convenience wrapper: windowed local resolution -> normalized per-atom weights.

  Returns a flex.double parallel to sites_cart, suitable for the ``weights``
  argument of the real-space refinement engine. See the module docstring.

  NOTE: the single-map windowed d99 estimate above is cheap but noisy at the
  per-atom scale. When two half maps are available, prefer the half-map FSC
  estimate below (fsc_local_resolution_weights) -- it is the validated,
  model-free local-resolution signal.
  """
  resolution = windowed_local_resolution(
    map_data=map_data, unit_cell=unit_cell, sites_cart=sites_cart,
    window_edge=window_edge)
  return weights_from_resolution(
    resolution, power=power, weight_min=weight_min, weight_max=weight_max)


def fsc_local_resolution(map_model_manager, box_cushion=5.0,
                         map_id_1='map_manager_1', map_id_2='map_manager_2',
                         fsc_cutoff=0.5, n_bins=20):
  """
  Per-atom local resolution (Angstrom) from half-map FSC.

  Uses the validated map_model_manager.local_resolution_map (band-pass each
  half map to resolution shells, take a smoothed local real-space correlation
  between them, and find where it crosses fsc_cutoff at each voxel), then
  samples that field at the atoms. The manager is boxed around the model first
  (extract_all_maps_around_model) so the per-shell FFTs run on a small grid --
  on a full-size EM map this is the difference between seconds and hours.

  Requires two half maps registered in the manager under map_id_1 / map_id_2.
  Returns a flex.double parallel to the model's atoms.

  fsc_cutoff defaults to 0.5 rather than the usual 0.143: validated against a
  synthetic ground truth, the higher cutoff is much more robust to the fine bias
  that a small analysis neighborhood introduces (spurious local noise
  correlation clears 0.143 easily but not 0.5), so it preserves local-resolution
  *contrast* -- which is what the weights depend on. The trade-off is that
  absolute values run coarser than a 0.143 estimate on real (gradual-falloff)
  maps.
  """
  b = map_model_manager.extract_all_maps_around_model(box_cushion=box_cushion)
  lr = b.local_resolution_map(
    map_id_1=map_id_1, map_id_2=map_id_2,
    fsc_cutoff=fsc_cutoff, n_bins=n_bins).map_data()
  xrs = b.model().get_xray_structure()
  sites_frac = xrs.unit_cell().fractionalize(xrs.sites_cart())
  return flex.double([maptbx.eight_point_interpolation(lr, sites_frac[i])
                      for i in range(sites_frac.size())])


def fsc_local_resolution_weights(map_model_manager, power=2.0,
                                 weight_min=0.1, weight_max=10.0, **kw):
  """
  Convenience wrapper: half-map FSC local resolution -> normalized per-atom
  weights (mean 1). Preferred over the windowed-d99 weights when half maps are
  available. Extra keyword arguments are passed to fsc_local_resolution.
  """
  resolution = fsc_local_resolution(map_model_manager, **kw)
  return weights_from_resolution(
    resolution, power=power, weight_min=weight_min, weight_max=weight_max)
