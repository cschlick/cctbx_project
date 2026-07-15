from __future__ import absolute_import, division, print_function
# LIBTBX_SET_DISPATCHER_NAME mmtbx.development.rsr_peratom

from iotbx.cli_parser import run_program
from mmtbx.programs import rsr_peratom

if __name__ == '__main__':
  run_program(program_class=rsr_peratom.Program)
