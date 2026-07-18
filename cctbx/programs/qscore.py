from __future__ import absolute_import, division, print_function
import json
from pathlib import Path

from cctbx.array_family import flex
from libtbx.program_template import ProgramTemplate
from libtbx import group_args
from cctbx.maptbx.qscore import (
    calc_qscore,
    write_bild_spheres,
)
from libtbx.utils import Sorry
import numpy as np

# =============================================================================


def _fmt(v, nd=2):
  """Format a scalar Q-score; None/NaN render as '--'."""
  if v is None or (isinstance(v, float) and np.isnan(v)):
    return "--"
  return ("%%.%df" % nd) % float(v)


class Program(ProgramTemplate):

  description = """
  Perform a Q-score analysis for map-model fit.

  Two probe-allocation methods are offered as first-class options via
  qscore.probe_allocation_method:
    progressive  - the original method; probes are proposed and rejected
                   iteratively and it matches the mapq reference to floating
                   point. Slower.
    precalculate - probes are pre-allocated and rejected once. Much faster,
                   but yields slightly different results. This is the default.

  Both are kept deliberately. At the command line, exposing both is the most
  useful choice; if this ambiguity is confusing in a GUI, the right fix is to
  make an opinionated choice at the GUI level (pick one, hide the other),
  not to remove a method here.
  """

  datatypes = ['phil', 'model', 'real_map']

  master_phil_str = """
  include scope cctbx.maptbx.qscore.master_phil_str
  """

  def validate(self):
    # test for sane parameters
    if not (4<=self.params.qscore.n_probes<=512):
      raise Sorry("Provide n_probe values in the range 4-512")

    if  not (4<=self.params.qscore.shell_radius_num<=128):
      raise Sorry("Provide shell_radius_num values in range 4-128")

    if self.params.qscore.probe_allocation_method not in (
        "progressive", "precalculate"):
      raise Sorry(
        "probe_allocation_method must be 'progressive' or 'precalculate'")



  def run(self):
    # set up logging
    log = open("qscore.log", "w")

    self.logger.register("log", log, atexit_send_to=None)

    # Name default outputs (e.g. the --json file) after the input model rather
    # than the generic "cctbx_program" prefix -> "<model>_qscore_000.json"
    # (falling back to "qscore"). Only set when the standard output scope is
    # present and the user has not chosen their own prefix.
    output = getattr(self.params, "output", None)
    if output is not None and getattr(output, "prefix", None) is None:
      model_name = self.data_manager.get_default_model_name()
      if model_name:
        import os
        base = os.path.splitext(os.path.basename(model_name))[0]
        output.prefix = "%s_qscore" % base
      else:
        output.prefix = "qscore"

    # get initial data
    mmm = self.data_manager.get_map_model_manager()

    # calculate shells
    self.shells = []
    # add a range of shells
    start = self.params.qscore.shell_radius_start
    stop = self.params.qscore.shell_radius_stop
    num = self.params.qscore.shell_radius_num
    shells = list(np.linspace(start,stop,num,endpoint=True))

    for shell in reversed(shells):
      self.shells.insert(0,shell)


    # ignore hydrogens
    model = mmm.model()
    model.add_crystal_symmetry_if_necessary()
    model = model.remove_hydrogens()

    # make mmm
    mmm.set_model(model,overwrite=True)

    # print output
    self._print("Running Q-score:")
    self._print("\nRadial shells (Radii used to place probes):")
    self._print(",".join([str(round(shell,2)) for shell in shells]))
    self._print("\n Number of probes per radial shell: "+str(self.params.qscore.n_probes))

    # run qscore
    qscore_result= calc_qscore(
        mmm,
        selection=self.params.qscore.selection,
        n_probes=self.params.qscore.n_probes,
        rtol=self.params.qscore.rtol,
        shells=self.shells,
        nproc=self.params.qscore.nproc,
        probe_allocation_method=self.params.qscore.probe_allocation_method,
        log=self.logger)


    self.result = group_args(**qscore_result)
    records = self.result.qscore_records
    method = self.result.probe_allocation_method

    if self.params.qscore.selection is not None:
      model = model.select(model.selection(self.params.qscore.selection))
    assert model.get_number_of_atoms() == len(records["id"])

    q = self.result.qscore_per_atom            # flex.double, per atom
    n_all = len(records["id"])
    q_all = round(flex.mean(q), 2) if n_all > 0 else None

    # Localized report: mean Q over a cctbx selection (default 'protein').
    # This is the only granularity in the report -- one flexible, user-driven
    # number rather than fixed main-chain/side-chain/per-residue tables.
    report_selection = self.params.qscore.report_selection
    q_sel = None
    n_sel = 0
    if report_selection:
      try:
        sel = model.selection(report_selection)
      except Exception as e:
        raise Sorry("Invalid report_selection '%s': %s" % (report_selection, e))
      n_sel = sel.count(True)
      if n_sel > 0:
        q_sel = round(flex.mean(q.select(sel)), 2)

    # ---- report (labeled with the method used) ----
    self._print("\nFinished running.\n")
    self._print("Q-score results  [probe allocation method: %s]" % method)
    self._print("  Q-score, all atoms : %s  (N = %d)" % (_fmt(q_all), n_all))
    if report_selection:
      if n_sel > 0:
        self._print("  Q-score, selection '%s' : %s  (N = %d)"
                    % (report_selection, _fmt(q_sel), n_sel))
      else:
        self._print("  Q-score, selection '%s' : no atoms matched"
                    % report_selection)

    # store in results
    self.result.q_score_overall = q_all
    self.result.q_score_selection = q_sel
    self.result.q_score_selection_string = report_selection
    self.result.q_score_selection_n = n_sel

    # write out
    if self.params.qscore.write_probes:
      self.write_bild_spheres()

    if self.params.qscore.write_to_bfactor_pdb:
      self.write_to_bfactor_pdb(model,self.result.qscore_per_atom)

    if self.params.qscore.write_qscore_mmcif:
      self.write_qscore_mmcif(model,self.result.qscore_per_atom)


  def get_results(self):
    return self.result

  def get_results_as_JSON(self):
    records = self.result.qscore_records
    n = len(records["id"])
    flat = []
    for i in range(n):
      row = {}
      for key, col in records.items():
        v = col[i]
        if isinstance(v, np.floating):
          v = float(v)
        elif isinstance(v, np.integer):
          v = int(v)
        row[key] = v
      flat.append(row)
    results_dict = {
      "probe_allocation_method": self.result.probe_allocation_method,
      "flat_results": flat,
    }
    return json.dumps(results_dict, indent=2)

  def write_to_bfactor_pdb(self,model,qscore_per_atom):
    model.set_b_iso(qscore_per_atom)

    with open("qscore_bfactor_field.pdb","w") as fh:
      fh.write(model.model_as_pdb())

  def write_qscore_mmcif(self, model, qscore_per_atom):
    """
    Write an mmCIF carrying the per-atom Q-score in a dedicated
    _atom_site.qscore column (does not overload the B-factor). The column is
    non-standard, so treat it as a working carrier, not a deposition field.
    """
    import iotbx.cif

    # crystal-symmetry block (if any), then the atom_site block -- same pattern
    # model_as_mmcif uses internally, so the file is complete and valid.
    cs = model.crystal_symmetry()
    cif_block = cs.as_cif_block() if cs is not None else None

    hier_block = model.get_hierarchy().as_cif_block()
    loop = hier_block.get_loop("_atom_site")
    n = len(loop["_atom_site.id"])
    assert n == len(qscore_per_atom), (
      "atom_site rows (%d) != qscore atoms (%d)" % (n, len(qscore_per_atom)))
    loop.add_columns(
      {"_atom_site.qscore": ["%.4f" % float(v) for v in qscore_per_atom]})
    hier_block.add_loop(loop)

    if cif_block is not None:
      cif_block.update(hier_block)
    else:
      cif_block = hier_block

    cif = iotbx.cif.model.cif()
    cif["qscore"] = cif_block
    filename = self.get_default_output_filename() + ".cif"
    with open(filename, "w") as fh:
      cif.show(out=fh)
    self._print("Wrote per-atom Q-score mmCIF: %s (_atom_site.qscore)" % filename)

  def write_bild_spheres(self):
    # write bild files
    if self.params.qscore.write_probes:
      self._print("Writing probe debug files...Using a small selection is recommended",
            file=self.logger)
      debug_path = Path("qscore_debug")
      debug_path.mkdir(exist_ok=True)
      for i,shell in enumerate(self.shells):
        shell = str(round(shell,2))
        probe_xyz = self.result.probe_xyz[i]
        n_shells, n_atoms,n_probes,_ = self.result.probe_xyz.shape
        probe_xyz_flat = probe_xyz.reshape((n_atoms*n_probes,3))
        out_file = Path(debug_path,"probes_shell_"+shell+".bild")
        write_bild_spheres(probe_xyz_flat,str(out_file),r=0.2)
