from __future__ import absolute_import, division, print_function
import json
from pathlib import Path

from cctbx.array_family import flex
from libtbx.program_template import ProgramTemplate
from libtbx import group_args
from cctbx.maptbx.qscore import (
    calc_qscore,
    group_indices_by_residue,
    write_bild_spheres,
)
from libtbx.utils import Sorry
import numpy as np

# =============================================================================


def _fmt(v, nd=2):
  """Format a scalar for a text table; None/NaN render as '--'."""
  if v is None or (isinstance(v, float) and np.isnan(v)):
    return "--"
  return ("%%.%df" % nd) % float(v)


def _safe_mean(values):
  """Mean of a possibly-empty array; empty -> NaN."""
  a = np.asarray(values, dtype=float)
  return float(np.mean(a)) if a.size else float("nan")


def _format_table(headers, rows):
  """Render an aligned fixed-width text table (pandas-free)."""
  cells = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
  widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
  out = []
  for r_i, row in enumerate(cells):
    out.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    if r_i == 0:
      out.append("  ".join("-" * widths[i] for i in range(len(headers))))
  return "\n".join(out)


def _group_indices_by_chain(records):
  groups = {}
  for i, c in enumerate(records["chain_id"]):
    groups.setdefault(c, []).append(i)
  return groups


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
    q_np = np.array([float(v) for v in q])

    # main-chain / side-chain per-atom masks (aligned with records order)
    sel_mc = model.selection(
      "protein and (name C or name N or name CA or name O or name CB)")
    mc = sel_mc.as_numpy_array()
    sc = ~mc

    q_mc = round(flex.mean(q.select(sel_mc)), 2) if sel_mc.count(True) > 0 else None
    q_sc = round(flex.mean(q.select(~sel_mc)), 2) if (~sel_mc).count(True) > 0 else None
    q_all = round(flex.mean(q), 2)

    # ---- report (pandas-free, labeled with the method used) ----
    self._print("\nFinished running.\n")
    self._print("Q-score results  [probe allocation method: %s]" % method)

    self._print("\nBy residue:")
    res_rows = []
    for idxs in group_indices_by_residue(records).values():
      i0 = idxs[0]
      idxs = np.array(idxs)
      res_rows.append([
        records["chain_id"][i0],
        records["resseq"][i0],
        records["resname"][i0],
        _fmt(records["Q-Residue"][i0]),
        _fmt(_safe_mean(q_np[idxs[mc[idxs]]])),
        _fmt(_safe_mean(q_np[idxs[sc[idxs]]])),
      ])
    self._print(_format_table(
      ["chain", "resseq", "resname", "Q-Residue", "Main Chain", "Side Chain"],
      res_rows))

    self._print("\nBy chain:")
    chain_rows = []
    chain_means = {}
    for chain_id, idxs in _group_indices_by_chain(records).items():
      m = float(np.mean(q_np[np.array(idxs)]))
      chain_means[chain_id] = round(m, 2)
      chain_rows.append([chain_id, len(idxs), _fmt(m)])
    self._print(_format_table(["chain", "N atoms", "Q-mean"], chain_rows))

    self._print("\nOverall:")
    self._print(_format_table(
      ["Main Chain", "Side Chain", "Overall"],
      [[_fmt(q_mc), _fmt(q_sc), _fmt(q_all)]]))

    # store in results
    self.result.q_score_chain_means = chain_means
    self.result.q_score_side_chain = q_sc
    self.result.q_score_main_chain = q_mc
    self.result.q_score_overall = q_all

    # write out
    if self.params.qscore.write_probes:
      self.write_bild_spheres()

    if self.params.qscore.write_to_bfactor_pdb:
      self.write_to_bfactor_pdb(model,self.result.qscore_per_atom)


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
