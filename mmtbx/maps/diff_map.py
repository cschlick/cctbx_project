"""
Model-vs-map real-space difference map engine: Fo - Fc_scaled, with local
box-wise Fourier-space k*exp(-B*s^2) scaling of the model map to the observed
map (mmtbx.bulk_solvent.complex_f_minus_f_kb_scaled).

Lifted out of mmtbx/command_line/real_space_diff_map.py so it can be shared as a
library. real_space_diff_map (which writes the sigma-normalised difference map)
and per-residue difference metrics (e.g. mmtbx.local_diff_z, which needs the RAW
residual to normalise by local noise instead) both call compdiff -- the only
difference is the sigma_scale flag.

sigma_scale=True (default) reproduces real_space_diff_map exactly: the result is
divided by its own sample standard deviation, so the map is in sigma units.
sigma_scale=False returns the raw scaled residual (Fo - Fc_scaled) and exposes
the divisor as .sd, leaving normalisation to the caller.
"""
from __future__ import absolute_import, division, print_function

from scitbx.array_family import flex
from cctbx import maptbx
from cctbx import miller
from cctbx import uctbx
from cctbx import crystal
from libtbx import adopt_init_args
import mmtbx.bulk_solvent
from six.moves import zip
from six.moves import range


def scale_k1(x, y):
  x = x.as_1d()
  y = y.as_1d()
  den = flex.sum(y*y)
  if(abs(den) < 1.e-9): return 0
  return flex.sum(x*y)/den


def scale_two_real_maps_in_fourier_space(m1, m2, cs, d_min, vector_map):
  f1 = maptbx.map_to_map_coefficients(m=m1, cs=cs, d_min=d_min)
  f2 = maptbx.map_to_map_coefficients(m=m2, cs=cs, d_min=d_min)
  if(vector_map):
    f2 = f2.phase_transfer(phase_source=f1)
  ss = 1./flex.pow2(f1.d_spacings().data()) / 4.
  bs = flex.double([i for i in range(0, 100)])
  mc = mmtbx.bulk_solvent.complex_f_minus_f_kb_scaled(
    f1.data(), f2.data(), bs, ss)
  crystal_gridding = maptbx.crystal_gridding(
    unit_cell             = cs.unit_cell(),
    space_group_info      = cs.space_group_info(),
    pre_determined_n_real = m1.all())
  fft_map = miller.fft_map(
    crystal_gridding     = crystal_gridding,
    fourier_coefficients = f1.array(data=mc))
  return fft_map.real_map_unpadded()


class compdiff(object):
  def __init__(
        self,
        map_data_obs,
        xrs,
        d_min,
        vector_map,
        box_dimension=30,
        sigma_scale=True):
    adopt_init_args(self, locals())
    self.crystal_gridding = maptbx.crystal_gridding(
      unit_cell             = self.xrs.unit_cell(),
      space_group_info      = self.xrs.space_group_info(),
      pre_determined_n_real = self.map_data_obs.all())
    self.n_real = self.crystal_gridding.n_real()
    crystal_gridding = maptbx.crystal_gridding(
      unit_cell             = self.xrs.unit_cell(),
      space_group_info      = self.xrs.space_group_info(),
      pre_determined_n_real = self.map_data_obs.all())
    mc = xrs.structure_factors(d_min=d_min).f_calc()
    fft_map = miller.fft_map(
      crystal_gridding     = crystal_gridding,
      fourier_coefficients = mc)
    fft_map.apply_sigma_scaling()
    self.map_data_calc = fft_map.real_map_unpadded()
    scale = scale_k1(x=self.map_data_obs, y=self.map_data_calc)
    self.map_data_calc = self.map_data_calc * scale
    #
    # result map
    self.sd = None
    self.map_result = flex.double(flex.grid(self.map_data_obs.all()))
    # iterate over boxes
    self.box_iterator()

  def box_iterator(self):
    p = self.xrs.unit_cell().parameters()
    b = maptbx.boxes_by_dimension(
      n_real = self.n_real,
      dim    = self.box_dimension,
      abc    = p[:3])
    i_box = 0
    for s, e in zip(b.starts, b.ends):
      i_box += 1
      map_box_obs  = maptbx.copy(self.map_data_obs,  s, e)
      map_box_calc = maptbx.copy(self.map_data_calc, s, e)
      map_box_obs.reshape(flex.grid(map_box_obs.all()))
      map_box_calc.reshape(flex.grid(map_box_calc.all()))
      #######
      # XXX Copy-paste from map_box
      abc = []
      for i in range(3):
        abc.append( p[i] * map_box_calc.all()[i]/self.n_real[i] )
      ucb = uctbx.unit_cell(
        parameters=(abc[0], abc[1], abc[2], p[3], p[4], p[5]))
      cs = crystal.symmetry(unit_cell=ucb, space_group="P1")
      #######
      diff_map = scale_two_real_maps_in_fourier_space(
        m1         = map_box_obs,
        m2         = map_box_calc,
        cs         = cs,
        d_min      = self.d_min,
        vector_map = self.vector_map)
      maptbx.set_box(
        map_data_from = diff_map,
        map_data_to   = self.map_result,
        start         = s,
        end           = e)
    sd = self.map_result.sample_standard_deviation()
    self.sd = sd
    if(self.sigma_scale and sd != 0):
      self.map_result = self.map_result/sd
