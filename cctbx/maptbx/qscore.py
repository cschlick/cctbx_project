from __future__ import division

from libtbx.utils import null_out
from cctbx.array_family import flex
from libtbx import easy_mp
import numpy as np
import numpy.ma as ma
from scipy.spatial import KDTree


master_phil_str = """
  qscore
  {

    nproc = 1
        .type = int
        .help = Number of processors to use
        .short_caption = Number of processors to use
        .expert_level = 1
    n_probes = 32
        .type = int
        .help = Number of radial probes to use
        .short_caption = Number of radial probes to use
        .expert_level = 1

    selection = None
      .type = str
      .help = Only calculate atoms within this selection
      .short_caption = Only test atoms within this selection
      .expert_level = 1

    report_selection = "protein"
      .type = str
      .help = "Localize the reported Q-score to this cctbx atom selection. The "
              "reported value is the mean Q-score over the matching atoms. This "
              "is the minimal way to localize the result; pass any selection "
              "string, e.g. 'chain A and resseq 50:80'. Set to None to report "
              "only the overall value."
      .short_caption = "Selection to localize the reported Q-score"

    shell_radius_start = 0.1
      .type = float
      .help = Start testing density at this radius from atom
      .short_caption = Start testing density at this radius from atom
      .expert_level = 1

    shell_radius_stop = 2
      .type = float
      .help = Stop testing density at this radius from atom
      .short_caption = Stop testing density at this radius from atom
      .expert_level = 1

    shell_radius_num = 20
      .type = int
      .help = The number of radial shells
      .short_caption = The number of radial shells (includes start/stop, so minimum 2)
      .expert_level = 1

    rtol = 0.9
      .type = float
      .help = Mapq rtol value, the "real" shell radii are r*rtol

    probe_allocation_method = *precalculate progressive
      .type = choice
      .help = "Method used to allocate radial probes. progressive is the original method where probes are proposed and rejected iteratively and matches the mapq reference implementation to floating point. precalculate pre-allocates probes and rejects them once; it is much faster but yields slightly different results. Both are retained on purpose."
      .short_caption = "Probe allocation method (progressive is paper-exact, precalculate is fast)"
      .expert_level = 1

    write_probes = False
      .type = bool
      .help = Write the qscore probes as a .bild file to visualize in Chimera

    write_to_bfactor_pdb = False
      .type = bool
      .help = Write out a pdb file with the Q-score per atom in the B-factor field

    write_qscore_mmcif = False
      .type = bool
      .help = "Write out an mmCIF file carrying the per-atom Q-score in a "
              "dedicated _atom_site.qscore column. Unlike write_to_bfactor_pdb "
              "this does not overload the B-factor, so the real ADP is "
              "preserved. Note the column is non-standard (not in the PDBx/mmCIF "
              "dictionary), so strict parsers or a deposition roundtrip may drop "
              "it; it is a working carrier for analysis, not a deposition field."
      .short_caption = "Write per-atom Q-score as an _atom_site.qscore mmCIF column"
  }

  """

################################################################################
#### Probe generation functions
################################################################################



# Fast numpy version
def generate_probes_np(sites_cart, rad, n_probes):
  """
  Generate probes using the same methodology as Pintile mapq, but vectorized

  sites_cart: np array of shape (n_atoms,3)
  rad: the radius at which to place the probes
  N: the number of probes per atom

  Returns:
    probes (np.ndarray): shape (n_atoms,n_probes,3)
  """
  assert sites_cart.ndim == 2 and sites_cart.shape[-1]==3, (
    "Provide coordinates in shape (n_atoms,3)")

  N = n_probes
  h = -1.0 + (2.0 * np.arange(N) / float(N-1))
  phis = np.arccos(h)

  thetas = np.zeros(N)
  a = (3.6 / np.sqrt(N * (1.0 - h[1:-1]**2)))
  thetas[1:-1] = a
  thetas = np.cumsum(thetas)


  x = np.sin(phis) * np.cos(thetas)
  y = np.sin(phis) * np.sin(thetas)
  z = np.cos(phis)

  probes = rad * np.stack([x, y, z], axis=-1)

  # Adjusting location of generated points relative to point ctr
  probes = probes.reshape(-1, 1, 3) + sites_cart.reshape(1, -1, 3)

  # reshape (n_atoms,n_probes,3)
  probes = probes.swapaxes(0,1)
  return probes


def get_probe_mask(
      atom_tree,
      probes_xyz,
      r=None,
      expected=None,
      log=null_out(),
      ):
  """
  sites_cart shape  (n_atoms,3)
  probes_xyz shape (n_atoms,n_probes,3)

  If expected is None, infer atom indices from probes_xyz
  Else expected should be a single value, or have shape  (n_atoms,n_probes)

  Restored from the pre-ac1fcf3a28 implementation. Note: this is called
  per-atom by shell_probes_progressive (n_atoms==1), so the `if not expected`
  falsy-zero path is harmless there (arange(1)==[0]==expected for atom 0).
  """

  assert r is not None, "Provide a radius"
  assert probes_xyz.ndim ==3 and probes_xyz.shape[-1] == 3,(
    "Provide probes_xyz as shape: (n_atoms,n_probes,3)")

  n_atoms_probe,n_probes,_ = probes_xyz.shape
  dim = probes_xyz.shape[-1] # 3 for cartesian coords


  # reshaped_probes shape (n_atoms*n_probes,3)
  reshaped_probes = probes_xyz.reshape(-1, 3)
  atom_indices = np.tile(np.arange(n_atoms_probe), (probes_xyz.shape[1], 1)).T

  if not expected:
    atom_indices = np.tile(np.arange(n_atoms_probe), (probes_xyz.shape[1], 1)).T
  else:
    atom_indices = np.full(probes_xyz.shape,expected)

  associated_indices = atom_indices.reshape(-1)


  # query
  # Check if any other tree points are within r of each query point
  query_points = reshaped_probes
  other_points_within_r = []
  for i, (query_point,idx) in enumerate(zip(query_points,associated_indices)):

    indices_within_r = atom_tree.query_ball_point(query_point, r)

    # Exclude the associated point
    associated_index = associated_indices[i]
    other_indices = []
    for  idx in indices_within_r:
      if idx != associated_index:
        other_indices.append(idx)
      if len(indices_within_r)==0:
        other_indices.append(-1)


    other_points_within_r.append(other_indices)

  # true are points that don't get rejected
  num_nbrs_other = np.array(
     [len(inds) for i,inds in enumerate(other_points_within_r)])

  num_nbrs_other = num_nbrs_other.reshape((n_atoms_probe,n_probes))
  mask = num_nbrs_other==0

  return mask


# Slow, paper-exact version (matches mapq to floating point)
def shell_probes_progressive(
      sites_cart=None,   # A numpy array of shape (N,3)
      atoms_tree=None,  # An atom_xyz scipy kdtree
      selection_bool=None,# Boolean atom selection
      n_probes=8,       # The desired number of probes per shell (maps to target)
      RAD=1.5,          # The nominal radius of this shell
      rtol=0.9,         # Multiplied with RAD to get actual radius
      log = null_out(),
      ):
  """
  Generate probes progressively for a single shell (radius).

  Restored from the pre-ac1fcf3a28 implementation. The original exposed
  n_probes_target / n_probes_max / n_probes_min separately; to share the
  current get_probes/GatherProbes contract (which passes a single n_probes),
  they are derived here as target=n_probes, max=2*n_probes, min=4. At the
  historical default n_probes=8 this reproduces the original (8, 16, 4) exactly.
  """
  # Derive the original triple from the single shared n_probes kwarg
  n_probes_target = n_probes
  n_probes_max = 2 * n_probes
  n_probes_min = 4

  # Do input validation
  if not atoms_tree:
    assert atoms_tree is None, (
      "If not providing an atom tree, \
        provide a 2d atom coordinate array to build tree")

    atoms_tree = KDTree(sites_cart)

  # Manage log
  if log is None:
    log = null_out()

  # manage selection input
  if selection_bool is None:
    selection_bool = np.full(len(sites_cart),True)
  selection_bool = np.asarray(selection_bool, dtype=bool)

  # Full-array indices of the selected atoms. The loop index below is
  # selection-local, but atoms_tree is over ALL atoms, so get_probe_mask must be
  # told the parent's full index sel_indices[atom_i] to exclude it correctly.
  sel_indices = np.where(selection_bool)[0]

  # do selection
  sites_cart_sel = sites_cart[selection_bool]
  n_atoms = sites_cart_sel.shape[0]

  all_pts = []  # list of probe arrays for each atom
  for atom_i in range(n_atoms):
    coord = sites_cart_sel[atom_i:atom_i+1]
    outRAD = RAD * rtol

    pts = []
    i_log = []
    # try to get at least numPts] points at [RAD] distance
    # from the atom, that are not closer to other atoms
    N_i = 50

    # If we find the necessary number of probes in the first iteration,
    #   then i will never go to 1
    for i in range(0, N_i):
      rejections = 0



      # progressively more points are grabbed  with each failed iter
      n_pts_to_grab = (n_probes_target + i * 2)

      # get the points in shape (n_atoms,n_pts_to_grab,3)
      outPts = generate_probes_np(coord, RAD, n_pts_to_grab)

      # initialize points to keep
      at_pts, at_pts_i = [None] * outPts.shape[1], 0

      # mask for outPts, are they are closest to the expected atom
      # mask shape (n_atoms,n_pts_to_grab)
      # NOTE: n_atoms != len(outPts)

      # will get mask of shape (n_atoms,n_probes)
      mask = get_probe_mask(atoms_tree,outPts,r=outRAD,expected=int(sel_indices[atom_i]),log=log)

      # identify which ones to keep, progressively grow pts list
      for pt_i, pt in enumerate(outPts[0]):
        keep = mask[0,pt_i] # only one atom TODO: vectorize atoms
        if keep:
          at_pts[at_pts_i] = pt
          at_pts_i += 1
        else:
          #print("REJECTING...",pt,file=log)
          rejections+=1
          pass

      # check if we have enough points to break the search loop
      if ( at_pts_i >= n_probes_target):
        pts.extend(at_pts[0:at_pts_i])
        pts = pts + [np.array([np.nan,np.nan,np.nan])]*(n_probes_max-len(pts))
        #print(pts)
        break

      i_log.append(i)
      if i>=N_i:
        assert False, "Too many iterations to get probes"
      # End sampling iteration



    #Finish working on a single atom
    pts = np.array(pts)  # should be shape (n_probes,3)
    if pts.shape == (0,): # all probes clashed, continue with zero probes
      pts = np.full((n_probes_max,3),np.nan)

    assert pts.shape == (n_probes_max,3),(
    f"Pts shape must be ({n_probes_max},3), not {pts.shape})")



    all_pts.append(pts)


  # prepare output
  probe_xyz = np.stack(all_pts)
  probe_mask = ~(np.isnan(probe_xyz))[:,:,0]

  return probe_xyz, probe_mask


################################################################################
#### Run shell functions for multiple shells(possibly in parallel)
################################################################################

class GatherProbes:
  def __init__(self,
               func,
               fixed_kwargs,
               ):
    self.func = func
    self.fixed_kwargs = fixed_kwargs

  def __call__(self,RAD):

    return self.func(RAD=RAD,**self.fixed_kwargs)


def get_probes(
    sites_cart=None,
    atoms_tree = None,
    shells = None,
    n_probes = None,
    rtol=None,
    nproc=1,
    selection_bool = None,
    worker_func=None,
    log = null_out()):

  """
  Generate probes for multiple radial shells (shells)
  """
  # Create before multiprocessing
  atoms_tree = KDTree(sites_cart)



  assert shells is not None, "Must provide explicit radial shells"
  fixed_kwargs = {
      "sites_cart": sites_cart,  # A numpy array of shape (N,3)
      "atoms_tree": atoms_tree,  # An atom_xyz scipy kdtree
      "selection_bool": selection_bool,  # Boolean atom selection
      "n_probes": n_probes,  # The desired number of probes per shell
      "rtol": rtol,  # Multiplied with RAD to get actual radius
      "log": log,
      }

  gather = GatherProbes(worker_func,fixed_kwargs)

  results = easy_mp.pool_map(
      processes=nproc,
      fixed_func=gather,
      args=shells)

  probe_xyz = [result[0] for result in results]
  probe_mask = [result[1] for result in results]

  n_shells = len(shells)
  probe_xyz_stacked = np.empty((n_shells,*probe_xyz[0].shape))
  for i,array in enumerate(probe_xyz):
    probe_xyz_stacked[i] = array
  return probe_xyz_stacked, np.array(probe_mask)


def shell_probes_precalculate(
      sites_cart=None,   # A numpy array of shape (N,3)
      atoms_tree=None,  # An atom_xyz scipy kdtree
      selection_bool=None,# Boolean atom selection
      n_probes=8,# The desired number of probes per shell
      RAD=1.5,          # The nominal radius of this shell
      rtol=0.9,         # Multiplied with RAD to get actual radius
      log = null_out(),
      strict = False,
      ):
  """
  Generate probes by precalculating for a single shell (radius)
  """

  # Do input validation
  if not atoms_tree:
    atoms_tree = KDTree(np.array(sites_cart))

  # Manage log
  if log is None:
    log = null_out()

  # manage selection input
  if selection_bool is None:
    selection_bool = np.full(len(sites_cart),True)
  selection_bool = np.asarray(selection_bool, dtype=bool)

  # Full-array indices of the selected atoms. Probes (and the rows below) are
  # built in selected order, but atoms_tree is over ALL atoms, so the parent
  # atom of the i-th selected atom is at full index sel_indices[i], not i.
  sel_indices = np.where(selection_bool)[0]

  # do selection
  sites_cart_sel = sites_cart[selection_bool]

  # get probe coordinates
  probe_xyz = generate_probes_np(sites_cart_sel, RAD, n_probes)
  n_atoms, n_probes, _ = probe_xyz.shape
  probe_xyz_flat = probe_xyz.reshape(-1,3)

  # modify "real" radius as in mapq
  outRAD = RAD*rtol

  # query kdtree to get neighbors and their distances
  dists, atom_indices = atoms_tree.query(probe_xyz_flat, k=2)
  dists = dists.reshape((n_atoms,n_probes,2))
  atom_indices = atom_indices.reshape((n_atoms,n_probes,2))

  # Build an index array that would be expected if each probe is near "its" atom
  # (the parent atom's index in the full atom array that atoms_tree was built on)
  row_indices = sel_indices[:, np.newaxis]

  # Mask for whether each probe's nearest atom is the one expected
  expected_atom_mask = atom_indices[:,:,0]==row_indices

  # A second mask to determine if the second nearest neighbor should be rejected
  #  (whether the second nearest neighbor is within the rejection radius)
  within_r_mask = dists[:,:,1]<outRAD #

  # Combine masks
  probe_mask = expected_atom_mask & ~within_r_mask

  # Debug/Validation on number of probes per atom
  n_probes_min = 4
  strict=False
  n_probes_per_atom = probe_mask.sum(axis=1)
  insufficient_probes = np.where(n_probes_per_atom<n_probes)[0]
  problematic_probes = np.where(n_probes_per_atom<n_probes_min)[0]
  if strict:
    if n_probes_per_atom.min() >= n_probes_min:
      print(
      f"Some atoms have less than {n_probes_min} probes. \
          ({len(problematic_probes)}). Consider raising n_probes",file=log)

  return probe_xyz, probe_mask


def calc_qscore(mmm,
                selection=None,
                shells=None,
                n_probes=8,
                rtol=0.9,
                nproc=1,
                probe_allocation_method="precalculate",
                log=null_out(),
                debug=False):
  """
  Calculate qscore from map model manager
  """
  model = mmm.model()
  # never do hydrogen
  model = model.remove_hydrogens()
  mmm.set_model(model)
  mm = mmm.map_manager()


  # Get atoms
  sites_cart = model.get_sites_cart().as_numpy_array()

  # do selection
  if selection != None:
    selection_bool = mmm.model().selection(selection) # boolean
    if selection_bool.count(True) ==0:
      print("Finished... nothing selected",file=log)
      return {"qscore_per_atom":None}
  else:
    selection_bool = flex.bool(model.get_number_of_atoms(),True)


  # determine worker func
  if probe_allocation_method == "progressive":
    worker_func = shell_probes_progressive
  elif probe_allocation_method == "precalculate":
    worker_func = shell_probes_precalculate
  else:
    raise ValueError(
      "probe_allocation_method must be 'progressive' or 'precalculate', "
      "got %r" % probe_allocation_method)


  # Get probes and probe mask (probes to reject)
  probe_xyz,probe_mask = get_probes(
    sites_cart=sites_cart,
    atoms_tree = None,
    shells=shells,
    n_probes=n_probes,
    rtol=rtol,
    nproc=nproc,
    selection_bool = selection_bool,
    worker_func=worker_func,
    log = log,
    )

  # after the probe generation, versions 1 and 2 are the same

  # infer params from shape
  n_shells, n_atoms, n_probes, _ = probe_xyz.shape

  # flatten
  probe_xyz_flat = probe_xyz.reshape((n_atoms * n_shells * n_probes, 3))
  probe_mask_flat = probe_mask.reshape(-1)  # (n_shells*n_atoms*n_probes,)
  masked_probe_xyz_flat = probe_xyz_flat[probe_mask_flat]

  # interpolate
  volume = mm.map_data().as_numpy_array()
  voxel_size = mm.pixel_sizes()
  masked_density = mm.density_at_sites_cart(
    flex.vec3_double(masked_probe_xyz_flat)).as_numpy_array()

  d_vals = np.full((n_shells, n_atoms, n_probes),np.nan)
  d_vals[probe_mask] = masked_density

  # g vals
  # create the reference data
  M = volume
  maxD = min(M.mean() + M.std() * 10, M.max())
  minD = max(M.mean() - M.std() * 1, M.min())
  A = maxD - minD
  B = minD
  u = 0
  sigma = 0.6
  x = np.array(shells)
  y = A * np.exp(-0.5 * ((x - u) / sigma) ** 2) + B

  # Stack and reshape data for correlation calc

  # stack the reference to shape (n_shells,n_atoms,n_probes)
  g_vals = np.repeat(y[:, None], n_probes, axis=1)
  g_vals = np.expand_dims(g_vals, 1)
  g_vals = np.tile(g_vals, (n_atoms, 1))

  # set masked area to nan
  g_vals[~probe_mask] = np.nan

  # reshape
  g_vals_2d = g_vals.transpose(1, 0, 2).reshape(g_vals.shape[1], -1)
  d_vals_2d = d_vals.transpose(1, 0, 2).reshape(d_vals.shape[1], -1)
  mask_2d = probe_mask.transpose(1, 0, 2).reshape(probe_mask.shape[1], -1)

  # CALCULATE Q
  q = rowwise_corrcoef(g_vals_2d, d_vals_2d, mask=mask_2d)

  # round sensibly
  q = np.around(q,4)

  # aggregate per residue (pandas-free)
  model = model.select(flex.bool(selection_bool))
  records = build_qscore_records(model, q)
  qscore_per_residue = flex.double(list(records["Q-Residue"]))
  q = flex.double([float(v) for v in q])

  # Output
  result = {
    "qscore_per_atom":q,
    "qscore_per_residue":qscore_per_residue,
    "qscore_records":records,
    "probe_allocation_method":probe_allocation_method,
    }

  if debug:
    # Collect debug data
    result.update({
      "atom_xyz":sites_cart,
      "probe_xyz":probe_xyz,
      "probe_mask":probe_mask,
      "d_vals":d_vals,
      "g_vals":g_vals,
    })

  return result


def rowwise_corrcoef(A, B, mask=None):
  """Numpy masked array rowwise correlation coefficient"""
  assert A.shape == B.shape, (
      f"A and B must have the same shape, got: {A.shape} and {B.shape}")

  if mask is not None:
    assert mask.shape == A.shape, "mask must have the same shape as A and B"
    A = ma.masked_array(A, mask=np.logical_not(mask))
    B = ma.masked_array(B, mask=np.logical_not(mask))

  # Calculate means
  A_mean = ma.mean(A, axis=1, keepdims=True)
  B_mean = ma.mean(B, axis=1, keepdims=True)

  # Subtract means
  A_centered = A - A_mean
  B_centered = B - B_mean

  # Calculate sum of products
  sumprod = ma.sum(A_centered * B_centered, axis=1)

  # Calculate square roots of the sum of squares
  sqrt_sos_A = ma.sqrt(ma.sum(A_centered**2, axis=1))
  sqrt_sos_B = ma.sqrt(ma.sum(B_centered**2, axis=1))

  # Return correlation coefficients
  cc = sumprod / (sqrt_sos_A * sqrt_sos_B)
  return cc.data



def group_indices_by_residue(records):
  """
  Group atom indices by residue, keyed on (chain_id, resseq, resname), in
  first-seen order. Pandas-free.

  Params:
    records (dict): dict-of-columns table from cctbx_atoms_to_records

  Returns:
    dict[tuple, list[int]]: residue key -> list of atom indices (row indices
      into every column of `records`)
  """
  groups = {}
  keys = zip(records["chain_id"], records["resseq"], records["resname"])
  for i, key in enumerate(keys):
    groups.setdefault(key, []).append(i)
  return groups


def build_qscore_records(model, qscore_per_atom):
  """
  Attach per-atom Q-score and a per-residue mean ("Q-Residue") to the atom
  table, without pandas. Residues are grouped by (chain_id, resseq, resname);
  the mean Q over a residue's non-hydrogen atoms is broadcast back to each of
  its atoms (matching the former pandas groupby-mean behaviour).

  Params:
    model: an mmtbx/iotbx model (already hydrogen-stripped and selected)
    qscore_per_atom: per-atom Q-scores, length == model.get_number_of_atoms()

  Returns:
    dict: cctbx_atoms_to_records columns plus "Q-score" and "Q-Residue"
  """
  records = cctbx_atoms_to_records(model.get_atoms())
  q = np.asarray([float(v) for v in qscore_per_atom], dtype=float)
  n = len(records["id"])
  assert q.shape[0] == n, (
    "qscore_per_atom length %d != n atoms %d" % (q.shape[0], n))
  records["Q-score"] = q

  q_residue = np.empty(n, dtype=float)
  for idxs in group_indices_by_residue(records).values():
    q_residue[idxs] = float(np.mean(q[idxs]))
  records["Q-Residue"] = q_residue
  return records


def write_bild_spheres(xyz,filename="sphere.bild",r=0.5):
  """
  Write a chimerax .bild file with spheres

  Params:
    xyz (np.array): Cartesian coordinates (N,3)
    filename (str): The filename for the .bild file
    r (float): sphere radius
  """
  out = ""
  for x,y,z in xyz:
    s = f".sphere {x} {y} {z} {r}\n"
    out+=s

  with open(filename,"w") as fh:
    fh.write(out)



def cctbx_atoms_to_records(atoms):
  """
  Build a plain dict-of-columns table from a cctbx shared atoms object.
  Pandas-free replacement for the former cctbx_atoms_to_df.

  Params:
    atoms (iotbx_pdb_hierarchy_ext.af_shared_atom): The atom array

  Returns:
    dict[str, list | np.ndarray]: parallel columns (length == n atoms):
      id, model_id, chain_id, resseq, resname, name, element, altloc are
      lists; x, y, z are float numpy arrays.
  """
  keys = ("id","model_id","chain_id","resseq","resname","name","element","altloc")
  cols = {k: [] for k in keys}
  for atom in atoms:
    ag = atom.parent()          # atom_group
    rg = ag.parent()            # residue_group
    chain = rg.parent()         # chain
    model = chain.parent()      # model
    cols["id"].append(atom.i_seq)
    cols["model_id"].append(model.id)
    cols["chain_id"].append(chain.id)
    cols["resseq"].append(rg.resseq_as_int())
    cols["resname"].append(rg.unique_resnames()[0])
    cols["name"].append(atom.name.strip())
    cols["element"].append(atom.element)
    cols["altloc"].append(ag.altloc)

  xyz = atoms.extract_xyz().as_numpy_array()
  cols["x"] = xyz[:,0]
  cols["y"] = xyz[:,1]
  cols["z"] = xyz[:,2]
  return cols
