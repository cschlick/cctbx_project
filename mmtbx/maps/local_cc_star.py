"""
Local (per-residue) map-model agreement with a half-map derived CC* ceiling.

This is the native cctbx equivalent of Servalcat's ``localcc`` "good scores".
For every residue it reports, from a local real-space window (all grid points
within ``radius`` of the residue's atoms):

  cc_mapmodel   local CC(full map, model-calculated map)   -- the fit
  cc_half       local CC(half map 1, half map 2)           -- the local map quality
  cc_star       sqrt(2*cc_half / (1 + cc_half))            -- the achievable ceiling
  cc_gap        cc_star - cc_mapmodel                      -- headroom to the ceiling

``cc_star`` is the local form of the Rosenthal & Henderson (2003) map-vs-true
ceiling / Karplus & Diederichs (2012) CC*: the correlation a *perfect* model
would achieve against a map of this local quality. It is the honest yardstick
for cc_mapmodel -- a residue with cc_mapmodel = 0.6 in a region whose ceiling is
0.62 fits essentially as well as the data allow, while the same 0.6 where the
ceiling is 0.95 is a genuine local misfit. cc_mapmodel exceeding cc_star is the
signature of fitting noise (overfitting).

The local window is a hard sphere of grid points around the residue's atoms
(``mmtbx.maps.correlation.from_map_map_atoms``); Servalcat uses a soft Gaussian
kernel of comparable width. The two windowings differ in softness but measure
the same quantity; values track closely and rank orderings agree.

Everything runs on iotbx.map_model_manager primitives -- no Servalcat, no
subprocess, no external environment.
"""
from __future__ import absolute_import, division, print_function

import math

from libtbx import group_args
from cctbx.array_family import flex
from mmtbx.maps import correlation


def cc_star_from_cc_half(cc_half):
  """Local CC* ceiling from the half-half CC.

  CC_full = 2 r / (1 + r) is the full-map self-consistency implied by a
  half-map CC of r; CC* = sqrt(CC_full) is the map-vs-true correlation, i.e.
  the highest CC_mapmodel a perfect model could reach here. Returns None for a
  missing input and clamps the (rare, noise-driven) negative-r domain to 0.
  """
  if cc_half is None:
    return None
  if cc_half <= -1.0:
    return 0.0
  inner = 2.0 * cc_half / (1.0 + cc_half)
  if inner <= 0.0:
    return 0.0
  return math.sqrt(inner)


def default_radius(resolution):
  """Local-window radius when the caller gives none.

  A few resolution-shells wide: wide enough for a stable correlation over a
  residue-sized neighbourhood, tight enough to stay local. Floored at 2.5 A so
  high-resolution maps still gather enough grid points."""
  return max(2.5, float(resolution))


def _iter_residues(hierarchy):
  """Yield (chain_id, resseq, resname, altloc_summary, i_seqs) per residue group.

  One entry per residue_group (all altlocs of a residue share a window). i_seqs
  index into the model's atoms / sites_cart."""
  for md in hierarchy.models():
    for chain in md.chains():
      for rg in chain.residue_groups():
        atoms = rg.atoms()
        i_seqs = atoms.extract_i_seq()
        if i_seqs.size() == 0:
          continue
        resnames = set(ag.resname.strip() for ag in rg.atom_groups())
        yield group_args(
          chain_id = chain.id.strip(),
          resseq   = rg.resseq.strip(),
          icode    = rg.icode.strip(),
          resname  = "/".join(sorted(resnames)),
          i_seqs   = i_seqs)


def per_residue_local_cc_star(
      map_data_full,
      model_map_data,
      unit_cell,
      model,
      map_data_half1 = None,
      map_data_half2 = None,
      radius         = 3.0,
      selection      = None):
  """Per-residue local cc_mapmodel / cc_half / cc_star / cc_gap.

  All map_data arrays must share gridding (same origin/all/focus) -- the caller
  is responsible for boxing/resampling them onto a common grid first (a
  map_model_manager after box_all_maps_around_model_and_shift_origin does this).

  Args:
    map_data_full   : flex.double map, the experimental/full map
    model_map_data  : flex.double map calculated from the model, same gridding
    unit_cell       : cctbx unit cell of the (boxed) frame
    model           : mmtbx.model.manager, sites consistent with the maps
    map_data_half1/2: optional half maps, same gridding (enable cc_half/cc_star)
    radius          : local-window radius in Angstrom
    selection       : optional bool flex selection over atoms to restrict scoring

  Returns a group_args with .residues (list of per-residue group_args) and the
  aggregate means .cc_mapmodel_mean / .cc_half_mean / .cc_star_mean, plus
  .n_over_ceiling (residues where cc_mapmodel exceeds cc_star -- overfitting).
  """
  have_halves = map_data_half1 is not None and map_data_half2 is not None
  correlation.assert_same_gridding(map_data_full, model_map_data)
  if have_halves:
    correlation.assert_same_gridding(map_data_full, map_data_half1)
    correlation.assert_same_gridding(map_data_full, map_data_half2)

  sites_cart = model.get_sites_cart()
  keep = None
  if selection is not None:
    keep = selection

  residues = []
  cc_mm_all = flex.double()
  cc_half_all = flex.double()
  cc_star_all = flex.double()
  n_over = 0

  for res in _iter_residues(model.get_hierarchy()):
    i_seqs = res.i_seqs
    if keep is not None:
      i_seqs = i_seqs.select(keep.select(i_seqs))
      if i_seqs.size() == 0:
        continue
    sites = sites_cart.select(i_seqs)

    cc_mm = correlation.from_map_map_atoms(
      map_1=map_data_full, map_2=model_map_data,
      sites_cart=sites, unit_cell=unit_cell, radius=radius)

    cc_half = cc_star = cc_gap = None
    if have_halves:
      cc_half = correlation.from_map_map_atoms(
        map_1=map_data_half1, map_2=map_data_half2,
        sites_cart=sites, unit_cell=unit_cell, radius=radius)
      cc_star = cc_star_from_cc_half(cc_half)
      if cc_star is not None:
        cc_gap = cc_star - cc_mm
        if cc_mm > cc_star:
          n_over += 1

    residues.append(group_args(
      chain_id=res.chain_id, resseq=res.resseq, icode=res.icode,
      resname=res.resname, n_atoms=i_seqs.size(),
      cc_mapmodel=cc_mm, cc_half=cc_half, cc_star=cc_star, cc_gap=cc_gap))

    cc_mm_all.append(cc_mm)
    if cc_half is not None:
      cc_half_all.append(cc_half)
      cc_star_all.append(cc_star)

  def _mean(a):
    return flex.mean(a) if a.size() else None

  return group_args(
    radius=radius,
    have_halves=have_halves,
    residues=residues,
    n_residues=len(residues),
    cc_mapmodel_mean=_mean(cc_mm_all),
    cc_half_mean=_mean(cc_half_all),
    cc_star_mean=_mean(cc_star_all),
    n_over_ceiling=n_over)
