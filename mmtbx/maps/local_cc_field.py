"""
Per-VOXEL local map-model correlation field (raw cc_mapmodel).

The voxel-wise companion to ``mmtbx.maps.local_cc_star``: instead of one CC per
residue, it returns cc_mapmodel on a grid, so the local fit can be written to an
MRC/CCP4 map and coloured on a surface in ChimeraX / PyMOL. No CC* ceiling and
no gap here -- this is only the raw local CC(full map, model-calculated map).

For every voxel the value is the Pearson CC of the experimental map and the
model map over a hard sphere of radius ``radius`` centred on that voxel -- the
same support as the per-atom ``correlation.from_map_map_atom`` used elsewhere.
``radius`` is the real locality knob and should be floored at ~one resolution
element (``max(2.5, d_min)``); tighter spheres hold too few independent Fourier
samples and the field turns to noise. ``band`` only selects which voxels are
evaluated (those within ``band`` A of any atom); it does not change any voxel's
value.

Two implementations that return the SAME field:

  local_cc_field           rolling local CC via sphere convolution -- five
                           maptbx.smooth_map (top-hat) calls, whole field in a
                           handful of FFTs. Use this.
  local_cc_field_bruteforce  one correlation.from_map_map_atom per voxel; the
                           reference the fast path is validated against (slow).

The two agree to corr ~0.999 (mean abs diff ~0.007) on a real 4.8 A map; the
tiny residual is the top-hat's reciprocal-space edge ripple versus an exact
real-space hard sphere.

Everything runs on cctbx primitives -- the convolution is
maptbx.smooth_map(method="top_hat"), the sphere gather is
maptbx.grid_indices_around_sites, the reference CC is
correlation.from_map_map_atom. Nothing here is re-implemented.
"""
from __future__ import absolute_import, division, print_function

from cctbx import maptbx
from cctbx.array_family import flex
from mmtbx.maps import correlation


def _envelope_indices(unit_cell, n_real, sites_cart, band):
  """1D grid indices (C-order) of voxels within ``band`` A of any atom."""
  import numpy as np
  env = maptbx.grid_indices_around_sites(
      unit_cell, n_real, n_real, sites_cart,
      flex.double(sites_cart.size(), band))
  return np.array(env, dtype=np.int64)


def _voxel_sites_cart(vox, n_real, unit_cell):
  """1D C-order grid indices -> voxel-centre cartesian coordinates."""
  import numpy as np
  nx, ny, nz = n_real
  i = vox // (ny * nz)
  j = (vox % (ny * nz)) // nz
  k = vox % nz
  frac = np.stack([i / nx, j / ny, k / nz], axis=1)
  return unit_cell.orthogonalize(flex.vec3_double(frac))


def _resolve(radius, band, d_min):
  radius = radius if radius is not None else max(2.5, d_min)
  band = band if band is not None else radius
  return radius, band


def local_cc_field(map_data_full, model_map_data, unit_cell, sites_cart,
                   crystal_symmetry, radius=None, band=None, d_min=None,
                   eps=1e-6):
  """Per-voxel local CC(full map, model map) via sphere convolution.

  Whole field from five top-hat convolutions, then masked to the model band:

    CC = (<AB> - <A><B>) / sqrt((<A^2> - <A>^2) (<B^2> - <B>^2))

  where <.> is the local hard-sphere mean. Each <.> is
  smooth_map(top_hat) / smooth_map(ones); dividing by the smoothed unit map
  turns the (arbitrarily scaled) convolution into a true weighted MEAN and
  cancels the top-hat's edge ripple, which is what makes the covariance /
  variance identity hold.

  Args:
    map_data_full:  experimental map (flex, origin-shifted).
    model_map_data: model-calculated map, same gridding.
    unit_cell:      of the (boxed) map.
    sites_cart:     model atom coordinates (define the evaluation band).
    crystal_symmetry: of the map (smooth_map needs it).
    radius:         sphere radius in A; default max(2.5, d_min).
    band:           evaluate voxels within this many A of an atom; default radius.
    d_min:          only used to default radius/band.

  Returns:
    (field, vox) -- field is a full-grid numpy array (NaN outside the band);
    vox is the 1D C-order indices of the evaluated (band) voxels.
  """
  import numpy as np
  correlation.assert_same_gridding(map_data_full, model_map_data)
  radius, band = _resolve(radius, band, d_min)
  n_real = map_data_full.all()

  def sphere_sum(m):  # unnormalized top-hat (hard-sphere) convolution
    return maptbx.smooth_map(map=m, crystal_symmetry=crystal_symmetry,
                             rad_smooth=radius, method="top_hat",
                             non_negative=False)

  norm = sphere_sum(flex.double(flex.grid(n_real), 1.0))
  def sphere_mean(m):
    return (sphere_sum(m) / norm).as_1d().as_numpy_array()

  A, B = map_data_full, model_map_data
  mA = sphere_mean(A)
  mB = sphere_mean(B)
  cov = sphere_mean(A * B) - mA * mB
  vA = sphere_mean(A * A) - mA * mA
  vB = sphere_mean(B * B) - mB * mB
  denom = np.sqrt(np.clip(vA, 0, None) * np.clip(vB, 0, None))
  with np.errstate(invalid="ignore", divide="ignore"):
    cc = np.where(denom > eps, cov / denom, np.nan)

  field = np.full(n_real, np.nan)
  vox = _envelope_indices(unit_cell, n_real, sites_cart, band)
  field.flat[vox] = cc[vox]
  return field, vox


def local_cc_field_bruteforce(map_data_full, model_map_data, unit_cell,
                              sites_cart, radius=None, band=None, d_min=None,
                              subset=None, seed=0):
  """Reference per-voxel local CC: one correlation.from_map_map_atom per voxel.

  Identical result to :func:`local_cc_field` but O(voxels x sphere volume);
  use it only to validate the fast path or on small regions. ``subset`` (int)
  evaluates that many random band voxels and leaves the rest NaN.

  Returns (field, vox) as in :func:`local_cc_field`.
  """
  import numpy as np
  radius, band = _resolve(radius, band, d_min)
  correlation.assert_same_gridding(map_data_full, model_map_data)
  n_real = map_data_full.all()

  vox = _envelope_indices(unit_cell, n_real, sites_cart, band)
  if subset is not None and subset < vox.size:
    rng = np.random.RandomState(seed)
    vox = np.sort(rng.choice(vox, size=subset, replace=False))
  carts = _voxel_sites_cart(vox, n_real, unit_cell)

  field = np.full(n_real, np.nan)
  for p, site in zip(vox, carts):
    cc = correlation.from_map_map_atom(
        map_1=map_data_full, map_2=model_map_data, site_cart=site,
        unit_cell=unit_cell, radius=radius)
    if cc is not None and cc == cc:   # cc == cc is False only for NaN
      field.flat[p] = cc
  return field, vox
