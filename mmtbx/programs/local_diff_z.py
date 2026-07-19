"""
mmtbx.local_diff_z -- native per-residue map-model *accuracy* metric: a
difference-density Z-score. The accuracy sibling of mmtbx.local_cc_star.

  phenix.local_diff_z model.cif full_map=emd.map \\
      half_map_1=half1.map half_map_2=half2.map resolution=3.2

Reuses the scaled Fo - Fc engine that phenix.real_space_diff_map uses
(mmtbx.maps.diff_map.compdiff) for the residual, then normalises by the LOCAL
half-map noise per residue. Positive diff_z = unexplained density (model too
small / misplaced); negative = model without map support (over-built). See
mmtbx/maps/local_diff_z.py for the definitions.
"""
from __future__ import absolute_import, division, print_function

import json

from libtbx.program_template import ProgramTemplate
from libtbx.utils import Sorry
from mmtbx.maps import local_diff_z as ldz
from mmtbx.maps.diff_map import compdiff

# =============================================================================

master_phil_str = """
local_diff_z {
  full_map = None
    .type = path
    .help = "Full (deposited) map. Defines the working frame."
  half_map_1 = None
    .type = path
    .help = "Half map 1 (required: the local noise denominator)."
  half_map_2 = None
    .type = path
    .help = "Half map 2 (required)."
  resolution = None
    .type = float
    .help = "Nominal resolution (A) for the model-calculated map. If unset, "
            "taken from the map_model_manager."
  radius = 2.0
    .type = float
    .help = "Local-window radius (A) for each residue."
  scattering_table = *electron n_gaussian wk1995 it1992 neutron
    .type = choice
    .help = "Scattering table for Fc. electron for cryo-EM."
  vector_map = False
    .type = bool
    .help = "True: (Fobs,Pobs)-(Fcalc,Pcalc) vector difference. False (default): "
            "(Fobs-Fcalc) with observed phases."
  selection = None
    .type = str
    .help = "Optional atom selection to restrict scoring."
  ignore_hydrogens = True
    .type = bool
    .help = "Drop hydrogens before scoring."
  json_out = None
    .type = path
    .help = "If set, write the full per-residue table as JSON here."
  write_bfactor_pdb = None
    .type = path
    .help = "If set, write the model with diff_z in the B-factor column."
}
"""


class Program(ProgramTemplate):

  description = """
mmtbx.local_diff_z -- per-residue difference-density Z-score (map-model accuracy).

Usage:
  phenix.local_diff_z model.cif local_diff_z.full_map=full.map \\
      local_diff_z.half_map_1=h1.map local_diff_z.half_map_2=h2.map \\
      local_diff_z.resolution=3.1

Per residue: net (Fo - Fc_scaled) over a local window, normalised by the local
half-map noise. diff_z > 0 = unexplained density, diff_z < 0 = over-built model.
"""

  datatypes = ['phil', 'model', 'real_map']
  master_phil_str = master_phil_str

  # ---------------------------------------------------------------------------
  def validate(self):
    p = self.params.local_diff_z
    if self.data_manager.get_default_model_name() is None:
      raise Sorry("A model (pdb/cif) is required.")
    if p.full_map is None:
      raise Sorry("local_diff_z.full_map is required.")
    if p.half_map_1 is None or p.half_map_2 is None:
      raise Sorry("Both half maps are required (they are the noise denominator).")

  # ---------------------------------------------------------------------------
  def run(self):
    p = self.params.local_diff_z
    mmm = self._build_frame(p)

    resolution = p.resolution or mmm.resolution()
    self._print("Working resolution : %.3f A" % resolution)
    self._print("Local-window radius: %.3f A" % p.radius)

    model = mmm.model()
    if p.ignore_hydrogens:
      model = model.remove_hydrogens()
      mmm.set_model(model, overwrite=True)
    model.setup_scattering_dictionaries(scattering_table=p.scattering_table)
    xrs = model.get_xray_structure()

    # scaled Fo - Fc_scaled residual (raw, not sigma-normalised) -- the SAME
    # engine phenix.real_space_diff_map uses.
    residual = compdiff(
      map_data_obs=mmm.map_manager().map_data(),
      xrs=xrs, d_min=resolution, vector_map=p.vector_map,
      sigma_scale=False).map_result

    noise = (mmm.get_map_manager_by_id('map_manager_1').map_data()
             - mmm.get_map_manager_by_id('map_manager_2').map_data())

    selection = None
    if p.selection is not None:
      selection = model.selection(p.selection)

    self.results = ldz.per_residue_diff_z(
      residual_map=residual, noise_map=noise,
      unit_cell=mmm.map_manager().crystal_symmetry().unit_cell(),
      model=model, radius=p.radius, selection=selection)

    self._report()

    if p.json_out:
      with open(p.json_out, 'w') as fh:
        json.dump(self.get_results_as_JSON(), fh, indent=2)
      self._print("Wrote %s" % p.json_out)
    if p.write_bfactor_pdb:
      self._write_bfactor_pdb(model, p.write_bfactor_pdb)

  # ---------------------------------------------------------------------------
  def _build_frame(self, p):
    """Load model + full + half maps into one boxed map_model_manager."""
    from iotbx.map_model_manager import map_model_manager
    dm = self.data_manager
    model = dm.get_model()

    def _rm(path):
      if path not in dm.get_real_map_names():
        dm.process_real_map_file(path)
      return dm.get_real_map(path)

    mm_full = _rm(p.full_map)
    mmm = map_model_manager(map_manager=mm_full, model=model.deep_copy())
    full_grid = tuple(mm_full.map_data().all())
    for path, map_id in [(p.half_map_1, 'map_manager_1'),
                         (p.half_map_2, 'map_manager_2')]:
      mm = _rm(path)
      if not self._same_cell(mm, mm_full):
        raise Sorry("Half map %s is in a different frame (unit cell) than the "
                    "full map; cannot form the noise map." % map_id)
      if tuple(mm.map_data().all()) != full_grid:
        self._print("  half map %s regridded %s -> %s"
                    % (map_id, tuple(mm.map_data().all()), full_grid))
        mm = mm.resample_on_different_grid(n_real=mm_full.map_data().all())
      mmm.add_map_manager_by_id(mm, map_id)

    resolution = p.resolution or mmm.resolution()
    mmm.box_all_maps_around_model_and_shift_origin(
      box_cushion=max(5.0, 2.0 * resolution))
    return mmm

  @staticmethod
  def _same_cell(a, b, tol=1e-2):
    pa = a.crystal_symmetry().unit_cell().parameters()
    pb = b.crystal_symmetry().unit_cell().parameters()
    return all(abs(x - y) <= tol * max(1.0, abs(x)) for x, y in zip(pa, pb))

  # ---------------------------------------------------------------------------
  def _report(self):
    r = self.results
    def f(v):
      return "  --  " if v is None else "%.3f" % v
    self._print("")
    self._print("=== mmtbx.local_diff_z (%d residues, radius %.2f A) ==="
                % (r.n_residues, r.radius))
    self._print("mean |diff_z|            : %s" % f(r.diff_z_mean_abs))
    self._print("residues |diff_z| > 3    : %d unexplained (+), %d over-built (-)"
                % (r.n_unexplained, r.n_excess))
    worst = sorted(r.residues, key=lambda x: abs(x.diff_z), reverse=True)[:10]
    self._print("\nLargest difference-density residues:")
    self._print("  chain resseq resname   diff_z   net_resid  interpretation")
    for x in worst:
      tag = "unexplained density" if x.diff_z > 0 else "over-built / no support"
      self._print("  %-5s %-6s %-7s  %+6.2f   %+8.4f  %s"
                  % (x.chain_id, x.resseq + x.icode, x.resname,
                     x.diff_z, x.net_residual, tag))

  # ---------------------------------------------------------------------------
  def _write_bfactor_pdb(self, model, path):
    r = self.results
    by_key = {(x.chain_id, x.resseq, x.icode): abs(x.diff_z) for x in r.residues}
    b = model.get_hierarchy().atoms().extract_b()
    for md in model.get_hierarchy().models():
      for chain in md.chains():
        for rg in chain.residue_groups():
          val = by_key.get((chain.id.strip(), rg.resseq.strip(), rg.icode.strip()))
          if val is not None:
            for atom in rg.atoms():
              b[atom.i_seq] = val
    model.set_b_iso(b)
    with open(path, 'w') as fh:
      fh.write(model.model_as_pdb())
    self._print("Wrote %s (|diff_z| in B-factor column)" % path)

  # ---------------------------------------------------------------------------
  def get_results(self):
    return self.results

  def get_results_as_JSON(self):
    r = self.results
    return dict(
      radius=r.radius, n_residues=r.n_residues,
      diff_z_mean_abs=r.diff_z_mean_abs,
      n_unexplained=r.n_unexplained, n_excess=r.n_excess,
      residues=[dict(chain_id=x.chain_id, resseq=x.resseq, icode=x.icode,
                     resname=x.resname, n_atoms=x.n_atoms, diff_z=x.diff_z,
                     net_residual=x.net_residual, noise_sigma=x.noise_sigma)
                for x in r.residues])
