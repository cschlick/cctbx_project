"""DEPRECATED shim -- this module was renamed to ``mmtbx.maps.local_c_ref``.

The local cryo-EM ceiling was called ``cc_star`` here, which collided with the
crystallographic Karplus & Diederichs CC* that cctbx already computes elsewhere
(``phenix.cc_star``, ``merging_stats.cc_star``). It now uses the cryo-EM
literature's own name, C_ref (Rosenthal & Henderson 2003).

Old names still work and map onto the new ones:

    per_residue_local_cc_star -> per_residue_local_c_ref
    cc_star_from_cc_half      -> c_ref_from_cc_half
    result.residues[i].cc_star / .c_ref   (both present)
    result.cc_star_mean       -> result.c_ref_mean

New code should import ``mmtbx.maps.local_c_ref``.
"""
from __future__ import absolute_import, division, print_function

from mmtbx.maps.local_c_ref import (  # noqa: F401
  c_ref_from_cc_half,
  default_radius,
  per_residue_local_c_ref,
)

#: Deprecated aliases.
cc_star_from_cc_half = c_ref_from_cc_half


def per_residue_local_cc_star(*args, **kwds):
  """Deprecated. Calls per_residue_local_c_ref and adds ``cc_star`` aliases so
  existing consumers that read ``.cc_star`` / ``.cc_star_mean`` keep working."""
  out = per_residue_local_c_ref(*args, **kwds)
  out.cc_star_mean = out.c_ref_mean
  for r in out.residues:
    r.cc_star = r.c_ref
  return out
