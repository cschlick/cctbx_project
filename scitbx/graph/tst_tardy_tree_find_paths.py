from scitbx.graph.tardy_tree import find_paths
from scitbx.graph import rigidity
from scitbx.graph import utils
import sys

def exercise_simple_loops(loop_size_max=8):
  for n_vertices in xrange(2, loop_size_max+1):
    edge_list = [tuple(sorted((i,(i+1)%n_vertices)))
      for i in xrange(n_vertices)]
    edge_sets = utils.construct_edge_sets(
      n_vertices=n_vertices, edge_list=edge_list)
    find_paths(edge_sets=edge_sets)

def three_archs_grow_edge_list(edge_list, offs, size):
  result = list(edge_list)
  i = 0
  for j in xrange(offs, offs+size):
    result.append((i,j))
    i = j
  result.append((1,i))
  return result

def exericse_three_archs(arch_size_max=8):
  for arch_size_1 in xrange(1, arch_size_max):
    edge_list_1 = three_archs_grow_edge_list(
      [], 2, arch_size_1)
    for arch_size_2 in xrange(1, arch_size_max):
      edge_list_12 = three_archs_grow_edge_list(
        edge_list_1, 2+arch_size_1, arch_size_2)
      for arch_size_3 in xrange(1, arch_size_max):
        n_vertices = 2 + arch_size_1 + arch_size_2 + arch_size_3
        edge_list_123 = three_archs_grow_edge_list(
          edge_list_12, 2+arch_size_1+arch_size_2, arch_size_3)
        es = utils.construct_edge_sets(
          n_vertices=n_vertices, edge_list=edge_list_123)
        bbes = utils.bond_bending_edge_sets(edge_sets=es)
        bbel = utils.extract_edge_list(edge_sets=bbes)
        dofs = [rigidity.determine_degrees_of_freedom(
          n_dim=3, n_vertices=n_vertices, edge_list=bbel, method=method)
            for method in ["float", "integer"]]
        assert dofs[0] == dofs[1]
        print " ", arch_size_1, arch_size_2, arch_size_3, dofs[0]-6
        print "edge_list:", edge_list_123
        expected = max(
          6,
          max(arch_size_1, arch_size_2, arch_size_3) + 1,
          arch_size_1 + arch_size_2 + arch_size_3 - 3)
        assert expected == dofs[0]
        is_rigid = dofs[0] == 6
        inferred_is_rigid = (
              arch_size_1 < 6
          and arch_size_2 < 6
          and arch_size_3 < 6
          and arch_size_1 + arch_size_2 + arch_size_3 < 10)
        assert inferred_is_rigid == is_rigid
        s = (arch_size_1 < 6) \
          + (arch_size_2 < 6) \
          + (arch_size_3 < 6)
        jps = find_paths(edge_sets=es)
        assert len(jps[1]) == s

def run(args):
  assert len(args) == 0
  exercise_simple_loops()
  exericse_three_archs()
  print "OK"

if (__name__ == "__main__"):
  run(sys.argv[1:])
