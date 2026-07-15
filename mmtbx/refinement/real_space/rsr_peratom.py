"""
Minimal per-atom real-space refinement (RSR) engine.

A small, UI-free re-assembly of the "modern" Phenix real-space refinement
idioms on top of the primitive cctbx building blocks. The runnable
Program-template wrapper lives in mmtbx/programs/rsr_peratom.py; this module
holds only the reusable engine so it can be imported and unit-tested on its own.

The actual coordinate minimization is delegated to the same engine Phenix uses,
mmtbx.refinement.real_space.individual_sites.easy, driven with the tricubic
interpolation gradients (the Phenix default).
"""
from __future__ import absolute_import, division, print_function

from libtbx.utils import null_out
from mmtbx.refinement.real_space import individual_sites


def run(model, map_data, gradients_method="tricubic", max_iterations=150,
        selection=None, weights=None, log=None):
  """
  Per-atom real-space refinement of ``model`` against ``map_data``.

  Parameters
  ----------
  model : mmtbx.model.manager
      Model with geometry restraints already built
      (``model.process(make_restraints=True)``).
  map_data : scitbx.array_family.flex.double
      Real-space map sampled on the model's unit-cell grid (typically
      ``map_model_manager.map_data()``).
  gradients_method : str
      Interpolation/gradient scheme: one of fd, linear, quadratic, tricubic.
      Defaults to ``tricubic`` to match Phenix's ``phenix.real_space_refine``.
  max_iterations : int
      LBFGS iteration cap.
  selection : flex.bool or None
      Atoms to refine; ``None`` refines all atoms.
  weights : flex.double or None
      Per-atom weights on the density term, parallel to the model's atoms.
      Emphasize the map fit where local density quality is high and lean on
      geometry where it is poor. Only used with an analytic gradients_method
      (linear/quadratic/tricubic), not with "fd". ``None`` means uniform
      weighting (identical to standard real-space refinement).

  Returns
  -------
  mmtbx.model.manager
      The same ``model`` object, sites updated in place, returned for chaining.
  """
  if log is None:
    log = null_out()
  if weights is not None:
    if gradients_method == "fd":
      # No runtime barrier: the fd path sums finite-difference gradients and
      # has no per-atom weight hook, so weights would be silently dropped.
      print("rsr_peratom: per-atom weights are ignored with "
            "gradients_method='fd'; use an analytic method (linear, quadratic, "
            "or tricubic) to apply them.", file=log)
    else:
      # The per-atom weights scale the density term, but the overall
      # density-vs-geometry weight is still determined without them, so the
      # weights act relatively rather than in absolute units.
      print("rsr_peratom: applying per-atom density weights; the overall "
            "density-vs-geometry weight is determined without them, so the "
            "weights act relatively.", file=log)
  refined = individual_sites.easy(
    map_data                    = map_data,
    xray_structure              = model.get_xray_structure(),
    pdb_hierarchy               = model.get_hierarchy(),
    geometry_restraints_manager = model.get_restraints_manager(),
    gradients_method            = gradients_method,
    selection                   = selection,
    weights                     = weights,
    max_iterations              = max_iterations,
    log                         = log)
  # Push the refined coordinates back onto the model so downstream writers
  # (data_manager.write_model_file, model_as_mmcif, ...) see the result.
  model.set_sites_cart(sites_cart = refined.xray_structure.sites_cart())
  return model
