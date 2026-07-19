"""
mmtbx.local_cc_star -- native per-residue map-model agreement with a half-map
derived CC* ceiling. The cctbx equivalent of Servalcat's ``localcc`` scores.

  phenix.local_cc_star model.cif full_map=emd.map \\
      half_map_1=half1.map half_map_2=half2.map resolution=3.2

Reports, per residue: local CC(map, model), local CC(half1, half2), the CC*
ceiling sqrt(2r/(1+r)) implied by the local half-map quality, and the gap
between them. See mmtbx/maps/local_cc_star.py for the definitions.
"""
from __future__ import absolute_import, division, print_function

import json

from libtbx.program_template import ProgramTemplate
from libtbx.utils import Sorry
from mmtbx.maps import local_cc_star as lcs

# =============================================================================

master_phil_str = """
local_cc_star {
  full_map = None
    .type = path
    .help = "Full (deposited) map. Defines the working frame."
  half_map_1 = None
    .type = path
    .help = "Half map 1. Provide with half_map_2 to get cc_half/cc_star."
  half_map_2 = None
    .type = path
    .help = "Half map 2."
  resolution = None
    .type = float
    .help = "Nominal resolution (A) for the model-calculated map. If unset, "
            "taken from the map_model_manager."
  radius = None
    .type = float
    .help = "Local-window radius (A) for each residue. Default: max(2.5, "
            "resolution)."
  scattering_table = *electron n_gaussian wk1995 it1992 neutron
    .type = choice
    .help = "Scattering table for the model-calculated map. electron for cryo-EM."
  selection = None
    .type = str
    .help = "Optional atom selection to restrict scoring."
  ignore_hydrogens = True
    .type = bool
    .help = "Drop hydrogens before scoring (they carry little map signal)."
  json_out = None
    .type = path
    .help = "If set, write the full per-residue table as JSON here."
  write_bfactor_pdb = None
    .type = path
    .help = "If set, write the model with cc_mapmodel in the B-factor column "
            "for map visualisation."
}
"""


class Program(ProgramTemplate):

  description = """
mmtbx.local_cc_star -- per-residue local map-model CC with a half-map CC* ceiling.

Usage:
  phenix.local_cc_star model.cif local_cc_star.full_map=full.map \\
      local_cc_star.half_map_1=h1.map local_cc_star.half_map_2=h2.map \\
      local_cc_star.resolution=3.2

Per residue it reports local CC(map,model), local CC(half1,half2), the CC*
ceiling implied by that local map quality, and the gap cc_star - cc_mapmodel
(negative gap = model correlates above the noise ceiling = overfitting).
"""

  datatypes = ['phil', 'model', 'real_map']
  master_phil_str = master_phil_str

  # ---------------------------------------------------------------------------
  def validate(self):
    p = self.params.local_cc_star
    if self.data_manager.get_default_model_name() is None:
      raise Sorry("A model (pdb/cif) is required.")
    if p.full_map is None:
      raise Sorry("local_cc_star.full_map is required.")
    if (p.half_map_1 is None) != (p.half_map_2 is None):
      raise Sorry("Provide both half maps or neither.")
    if p.half_map_1 is None:
      self._print("WARNING: no half maps -> cc_half/cc_star unavailable; only "
                  "cc_mapmodel will be reported.")

  # ---------------------------------------------------------------------------
  def run(self):
    p = self.params.local_cc_star
    mmm, have_halves = self._build_frame(p)

    resolution = p.resolution or mmm.resolution()
    radius = p.radius or lcs.default_radius(resolution)
    self._print("Working resolution : %.3f A" % resolution)
    self._print("Local-window radius: %.3f A" % radius)

    model = mmm.model()
    if p.ignore_hydrogens:
      model = model.remove_hydrogens()
      mmm.set_model(model, overwrite=True)

    # Model-calculated map on the shared (boxed) grid.
    mmm.generate_map(model=model, d_min=resolution, map_id='model_map')

    def _md(map_id):
      mm = mmm.get_map_manager_by_id(map_id)
      return mm.map_data() if mm is not None else None

    selection = None
    if p.selection is not None:
      selection = model.selection(p.selection)

    self.results = lcs.per_residue_local_cc_star(
      map_data_full=_md('map_manager'),
      model_map_data=_md('model_map'),
      unit_cell=mmm.map_manager().crystal_symmetry().unit_cell(),
      model=model,
      map_data_half1=_md('map_manager_1') if have_halves else None,
      map_data_half2=_md('map_manager_2') if have_halves else None,
      radius=radius,
      selection=selection)

    self._report()

    if p.json_out:
      with open(p.json_out, 'w') as fh:
        json.dump(self.get_results_as_JSON(), fh, indent=2)
      self._print("Wrote %s" % p.json_out)

    if p.write_bfactor_pdb:
      self._write_bfactor_pdb(model, p.write_bfactor_pdb)

  # ---------------------------------------------------------------------------
  def _build_frame(self, p):
    """Load model + maps into one map_model_manager and unify the frame.

    Boxes every map around the model and origin-shifts so model and all maps
    share one gridding (a hard requirement for the local-CC primitive). Half
    maps whose unit cell differs from the full map (true frame disagreement)
    are dropped; half maps on a different grid of the same cell are resampled.
    """
    from iotbx.map_model_manager import map_model_manager
    dm = self.data_manager
    model = dm.get_model()

    def _rm(path):
      if path not in dm.get_real_map_names():
        dm.process_real_map_file(path)
      return dm.get_real_map(path)

    mm_full = _rm(p.full_map)
    mmm = map_model_manager(map_manager=mm_full, model=model.deep_copy())
    try:
      mmm.set_scattering_table(p.scattering_table)
    except Exception:
      pass

    have_halves = False
    if p.half_map_1 is not None:
      full_grid = tuple(mm_full.map_data().all())
      pending, ok = [], True
      for path, map_id in [(p.half_map_1, 'map_manager_1'),
                           (p.half_map_2, 'map_manager_2')]:
        mm = _rm(path)
        if not self._same_cell(mm, mm_full):
          self._print("  WARNING: half map %s unit cell differs from full map "
                      "-> frame mismatch; cc_half/cc_star unavailable." % map_id)
          ok = False
          break
        if tuple(mm.map_data().all()) != full_grid:
          self._print("  half map %s regridded %s -> %s"
                      % (map_id, tuple(mm.map_data().all()), full_grid))
          mm = mm.resample_on_different_grid(n_real=mm_full.map_data().all())
        pending.append((mm, map_id))
      if ok:
        for mm, map_id in pending:
          mmm.add_map_manager_by_id(mm, map_id)
        have_halves = True

    resolution = p.resolution or mmm.resolution()
    cushion = max(5.0, 2.0 * resolution)
    mmm.box_all_maps_around_model_and_shift_origin(box_cushion=cushion)
    return mmm, have_halves

  @staticmethod
  def _same_cell(a, b, tol=1e-2):
    pa = a.crystal_symmetry().unit_cell().parameters()
    pb = b.crystal_symmetry().unit_cell().parameters()
    return all(abs(x - y) <= tol * max(1.0, abs(x)) for x, y in zip(pa, pb))

  # ---------------------------------------------------------------------------
  def _report(self):
    r = self.results
    def f(v):
      return "  --  " if v is None else "%.4f" % v
    self._print("")
    self._print("=== mmtbx.local_cc_star (%d residues, radius %.2f A) ==="
                % (r.n_residues, r.radius))
    self._print("mean cc_mapmodel : %s" % f(r.cc_mapmodel_mean))
    if r.have_halves:
      self._print("mean cc_half     : %s" % f(r.cc_half_mean))
      self._print("mean cc_star     : %s   (local ceiling)" % f(r.cc_star_mean))
      self._print("residues over ceiling (cc_mapmodel > cc_star): %d / %d%s"
                  % (r.n_over_ceiling, r.n_residues,
                     "  -- possible overfitting" if r.n_over_ceiling else ""))
    # worst-fitting residues by gap to ceiling (or by cc_mapmodel if no halves)
    self._print("\nWorst-fitting residues:")
    if r.have_halves:
      worst = sorted([x for x in r.residues if x.cc_gap is not None],
                     key=lambda x: x.cc_gap, reverse=True)[:10]
      self._print("  chain resseq resname   cc_mm   cc_half  cc_star   gap")
      for x in worst:
        self._print("  %-5s %-6s %-7s  %6.3f  %6.3f   %6.3f  %+6.3f"
                    % (x.chain_id, x.resseq + x.icode, x.resname,
                       x.cc_mapmodel, x.cc_half, x.cc_star, x.cc_gap))
    else:
      worst = sorted(r.residues, key=lambda x: x.cc_mapmodel)[:10]
      self._print("  chain resseq resname   cc_mm")
      for x in worst:
        self._print("  %-5s %-6s %-7s  %6.3f"
                    % (x.chain_id, x.resseq + x.icode, x.resname, x.cc_mapmodel))

  # ---------------------------------------------------------------------------
  def _write_bfactor_pdb(self, model, path):
    r = self.results
    # map residue-group key -> cc_mapmodel, then stamp every atom in the group
    by_key = {}
    for x in r.residues:
      by_key[(x.chain_id, x.resseq, x.icode)] = x.cc_mapmodel
    b = model.get_hierarchy().atoms().extract_b()
    i = 0
    for md in model.get_hierarchy().models():
      for chain in md.chains():
        for rg in chain.residue_groups():
          key = (chain.id.strip(), rg.resseq.strip(), rg.icode.strip())
          val = by_key.get(key)
          for atom in rg.atoms():
            if val is not None:
              b[atom.i_seq] = val
      i += 1
    model.set_b_iso(b)
    with open(path, 'w') as fh:
      fh.write(model.model_as_pdb())
    self._print("Wrote %s (cc_mapmodel in B-factor column)" % path)

  # ---------------------------------------------------------------------------
  def get_results(self):
    return self.results

  def get_results_as_JSON(self):
    r = self.results
    return dict(
      radius=r.radius,
      have_halves=r.have_halves,
      n_residues=r.n_residues,
      cc_mapmodel_mean=r.cc_mapmodel_mean,
      cc_half_mean=r.cc_half_mean,
      cc_star_mean=r.cc_star_mean,
      n_over_ceiling=r.n_over_ceiling,
      residues=[dict(chain_id=x.chain_id, resseq=x.resseq, icode=x.icode,
                     resname=x.resname, n_atoms=x.n_atoms,
                     cc_mapmodel=x.cc_mapmodel, cc_half=x.cc_half,
                     cc_star=x.cc_star, cc_gap=x.cc_gap)
                for x in r.residues])
