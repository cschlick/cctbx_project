from __future__ import absolute_import, division, print_function
# LIBTBX_SET_DISPATCHER_NAME mmtbx.local_c_ref

from iotbx.cli_parser import run_program
from mmtbx.programs import local_c_ref

if __name__ == '__main__':
  run_program(program_class=local_c_ref.Program)
