"""Minimal per-atom real-space refinement, Program-template wrapper."""
from __future__ import absolute_import, division, print_function
try:
  from phenix.program_template import ProgramTemplate
except ImportError:
  from libtbx.program_template import ProgramTemplate
import os
from libtbx.utils import Sorry
from mmtbx.refinement.real_space import rsr_peratom

# =============================================================================

master_phil_str = '''
gradients_method = fd linear quadratic *tricubic
  .type = choice(multi = False)
  .help = "Map interpolation / gradient scheme. Default (tricubic) matches phenix.real_space_refine."
max_iterations = 150
  .type = int
  .help = "LBFGS iteration cap per refinement run."
weighting = *none local_resolution
  .type = choice(multi = False)
  .help = "Per-atom weighting of the density term. 'none' is standard "
          "uniform-weight refinement. 'local_resolution' down-weights atoms in "
          "poor local density (estimated by a windowed d99) so they lean on the "
          "geometry restraints, and trusts the map where local resolution is "
          "high. Only applies to analytic gradients_method (not fd)."
local_resolution {
  window = 10.0
    .type = float
    .help = "Edge length (Angstrom) of the cubic window used for the per-atom "
            "windowed-d99 local-resolution estimate. Roughly 2-4x the map "
            "resolution is sensible."
  power = 2.0
    .type = float
    .help = "Sharpness of the resolution-to-weight mapping: "
            "weight = (median_resolution / local_resolution)**power, clamped "
            "and normalized to mean 1."
}
'''
# Note: output.file_name / output.overwrite come from ProgramTemplate's
# reserved output scope, so they are not redefined here.

# =============================================================================

class Program(ProgramTemplate):

  description = '''
Minimal per-atom real-space refinement.

Refines atomic coordinates against a real-space map using geometry restraints,
delegating to the same engine as phenix.real_space_refine (tricubic gradients
by default).

Minimum required inputs:
  Model file (.pdb / .cif)
  Real-space map file (.ccp4 / .mrc / .map)

How to run:
  mmtbx.rsr_peratom model.pdb map.ccp4
'''

  datatypes = ['model', 'real_map', 'phil']
  master_phil_str = master_phil_str

  # ---------------------------------------------------------------------------
  def validate(self):
    print('Validating inputs', file=self.logger)
    self.data_manager.has_models(raise_sorry=True)
    if not self.data_manager.has_real_maps():
      raise Sorry("Supply a real-space map file (.ccp4 / .mrc / .map).")

  # ---------------------------------------------------------------------------
  def run(self):
    # 1. Load model + map together on a common grid (map_model_manager idiom).
    #    Capture the input names up front: get_map_model_manager mutates the
    #    DataManager's default-model state, so query it before that call.
    model_name = self.data_manager.get_default_model_name()
    map_name = self.data_manager.get_default_real_map_name()
    print('Using model: %s' % model_name, file=self.logger)
    print('Using map:   %s' % map_name, file=self.logger)
    mmm = self.data_manager.get_map_model_manager(
      model_file = model_name,
      map_files  = map_name)
    model = mmm.model()

    # 2. Build geometry restraints if the model doesn't already carry them.
    if not model.restraints_manager_available():
      print('Building geometry restraints', file=self.logger)
      model.process(make_restraints=True)

    # 3. Optional per-atom weights from local map resolution.
    weights = None
    if self.params.weighting == 'local_resolution':
      if self.params.gradients_method == 'fd':
        print('weighting=local_resolution has no effect with '
              'gradients_method=fd; skipping.', file=self.logger)
      else:
        from mmtbx.refinement.real_space import local_resolution_weights
        print('Computing per-atom weights from local resolution '
              '(windowed d99, window=%.1f A)' %
              self.params.local_resolution.window, file=self.logger)
        weights = local_resolution_weights.local_resolution_weights(
          map_data    = mmm.map_data(),
          unit_cell   = model.crystal_symmetry().unit_cell(),
          sites_cart  = model.get_sites_cart(),
          window_edge = self.params.local_resolution.window,
          power       = self.params.local_resolution.power)

    # 4. Per-atom real-space refinement (tricubic by default).
    print('Refining (gradients_method=%s, max_iterations=%d)' %
      (self.params.gradients_method, self.params.max_iterations),
      file=self.logger)
    rsr_peratom.run(
      model            = model,
      map_data         = mmm.map_data(),
      gradients_method = self.params.gradients_method,
      max_iterations   = self.params.max_iterations,
      weights          = weights,
      log              = self.logger)
    self.model = model

    # 5. Write the refined model, named after the input model as
    #    "<model>_rsr.<ext>" (input format preserved). Passing the model object
    #    lets write_model_file keep the input pdb/cif format.
    if self.params.output.file_name is not None:
      filename = self.params.output.file_name
    else:
      stem, ext = os.path.splitext(os.path.basename(model_name))
      if ext.lower() not in ('.pdb', '.cif', '.ent', '.mmcif'):
        ext = '.cif'
      filename = '%s_rsr%s' % (stem, ext)
    written = self.data_manager.write_model_file(
      model,
      filename  = filename,
      overwrite = self.params.output.overwrite)
    print('Wrote refined model: %s' % written, file=self.logger)
    self.output_file = written

  # ---------------------------------------------------------------------------
  def get_results(self):
    from libtbx import group_args
    return group_args(
      model       = self.model,
      output_file = getattr(self, 'output_file', None))
