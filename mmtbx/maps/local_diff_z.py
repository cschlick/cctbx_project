"""
Local (per-residue) map-model *accuracy* metric: a difference-density Z-score.

Where local_cc_star measures agreement/precision (does the model correlate with
the map here), this measures accuracy (is there density the model fails to
explain, or model where there is no density). It is built on the same scaled
Fo - Fc residual that phenix.real_space_diff_map writes (mmtbx.maps.diff_map),
but instead of normalising by the whole-map sigma it normalises by the LOCAL
noise estimated from the half maps -- which is what decouples the signal from
local density strength (see note below).

Per residue, from a local window (grid points within ``radius`` of the residue's
atoms):

  net_residual  = mean(Fo - Fc_scaled)          # signed: + missing density, - excess model
  noise_sigma   = rms(half1 - half2) / 2         # local full-map noise sigma (see below)
  diff_z        = net_residual / noise_sigma     # signed difference-to-noise ratio

diff_z > 0 means unexplained density on the residue (model too small / misplaced /
a missing group); diff_z < 0 means model density with no map support (over-built /
wrong rotamer or register). |diff_z| ranks how badly the local density disagrees
with the model, in units of the local map noise.

Why local, half-map noise (not whole-map sigma): the raw residual magnitude
scales with local density strength, so a whole-map-sigma normalisation makes
strong, well-fit regions look worst. The half-map difference (h1 - h2) is pure
noise; its local rms is the local noise level, so dividing by it removes the
density-strength confound and leaves a genuine accuracy signal. (h1 and h2 are
independent halves, so var(h1-h2) = 2*var(half) and the full map = (h1+h2)/2 has
noise sigma = rms(h1-h2)/2.)

This is the *accuracy* sibling of mmtbx.maps.local_cc_star (precision); the two
share inputs (model + full + half maps), frame handling, and the per-residue
windowing primitive (mmtbx.maps.correlation.grid_indices_around_sites).
"""
from __future__ import absolute_import, division, print_function

import math

from libtbx import group_args
from cctbx import maptbx
from cctbx.array_family import flex


def _iter_residues(hierarchy):
  """Yield per-residue-group info: chain_id, resseq, icode, resname, i_seqs."""
  for md in hierarchy.models():
    for chain in md.chains():
      for rg in chain.residue_groups():
        i_seqs = rg.atoms().extract_i_seq()
        if i_seqs.size() == 0:
          continue
        resnames = set(ag.resname.strip() for ag in rg.atom_groups())
        yield group_args(
          chain_id=chain.id.strip(), resseq=rg.resseq.strip(),
          icode=rg.icode.strip(), resname="/".join(sorted(resnames)),
          i_seqs=i_seqs)


def per_residue_diff_z(
      residual_map,
      noise_map,
      unit_cell,
      model,
      radius   = 2.0,
      selection= None):
  """Per-residue signed difference-density Z-score.

  Args:
    residual_map : flex.double, RAW scaled Fo - Fc_scaled (mmtbx.maps.diff_map
                   compdiff with sigma_scale=False), on the working grid
    noise_map    : flex.double, half1 - half2 on the SAME grid
    unit_cell    : cctbx unit cell of the (boxed) frame
    model        : mmtbx.model.manager, sites consistent with the maps
    radius       : local-window radius in Angstrom
    selection    : optional bool flex selection over atoms to restrict scoring

  Returns a group_args with .residues (per-residue group_args) and aggregates
  .diff_z_mean_abs, .n_unexplained (diff_z > +3), .n_excess (diff_z < -3).
  """
  from mmtbx.maps import correlation  # for assert_same_gridding
  correlation.assert_same_gridding(residual_map, noise_map)

  sites_cart = model.get_sites_cart()
  keep = selection
  fft_n, fft_m = residual_map.focus(), residual_map.all()

  residues = []
  z_abs = flex.double()
  n_unexplained = n_excess = 0

  for res in _iter_residues(model.get_hierarchy()):
    i_seqs = res.i_seqs
    if keep is not None:
      i_seqs = i_seqs.select(keep.select(i_seqs))
      if i_seqs.size() == 0:
        continue
    sel = maptbx.grid_indices_around_sites(
      unit_cell=unit_cell, fft_n_real=fft_n, fft_m_real=fft_m,
      sites_cart=sites_cart.select(i_seqs),
      site_radii=flex.double(i_seqs.size(), radius))
    if sel.size() == 0:
      continue
    r = residual_map.select(sel)
    n = noise_map.select(sel)
    net = flex.mean(r)                       # signed net residual
    noise_sigma = math.sqrt(flex.mean(n * n)) / 2.0
    if noise_sigma < 1e-9:
      continue
    diff_z = net / noise_sigma

    residues.append(group_args(
      chain_id=res.chain_id, resseq=res.resseq, icode=res.icode,
      resname=res.resname, n_atoms=i_seqs.size(),
      net_residual=net, noise_sigma=noise_sigma, diff_z=diff_z))
    z_abs.append(abs(diff_z))
    if diff_z > 3.0:  n_unexplained += 1
    if diff_z < -3.0: n_excess += 1

  return group_args(
    radius=radius,
    residues=residues,
    n_residues=len(residues),
    diff_z_mean_abs=(flex.mean(z_abs) if z_abs.size() else None),
    n_unexplained=n_unexplained,
    n_excess=n_excess)
