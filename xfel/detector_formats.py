from __future__ import division
import time

def reverse_timestamp(timestamp):
  """Reverse of the xfel.cxi.cspad_ana.cspad_tbx.evt_timestamp()
  function.  From a string representation of a timestamp, @p
  timestamp, return the Unix time as a tuple of seconds and
  milliseconds.

  @param timestamp Human-readable ISO 8601 timestamp in string
                   representation
  @return          Tuple of the Unix time in seconds and milliseconds
  """

  tokens = timestamp.split('.')
  gmtime_tuple = time.strptime(tokens[0], '%Y-%m-%dT%H:%MZ%S')
  return (time.mktime(gmtime_tuple), float(tokens[1]))

def _get_detector_format_version_dict():
  from calendar import timegm
  from time import strptime

  # Note: one must take daylight savings into account when defining
  # the cutoff-times for the LCLS runs.  The last shift of a run
  # generally ends at 09:00 local time the day after the last day of
  # the run.
  f = '%Y-%m-%d, %H:%M %Z'

  # We have no metrology for the first two LCLS runs
  # LCLS started operation Oct 1, 2009
  #timegm(strptime('2009-10-01, 02:00 UTC', f))
  # LCLS run 1: until Dec 17, 2009
  #timegm(strptime('2009-12-18, 01:00 UTC', f)):
  # LCLS run 2: until Sep 15, 2010
  #timegm(strptime('2010-09-16, 02:00 UTC', f)):

  return {
    'Sacla.MPCCD': {
      'address':'Sacla.MPCCD',
      'start_time':None,
      'end_time':None
    },
    # LCLS run 3: until Mar 8, 2011
    #
    # 'CXI 3.1' corresponds to a quirky, old, and deprecated version
    # of the cctbx.xfel pickle format.
    'CXI 3.2': {
      'address':'CxiDs1-0|Cspad-0',
      'start_time':timegm(strptime('2010-09-16, 02:00 UTC', f)),
      'end_time'  :timegm(strptime('2011-03-09, 02:00 UTC', f))
    },
    # LCLS run 4: until Oct 28, 2011
    'CXI 4.1': {
      'address':'CxiDs1-0|Cspad-0',
      'start_time':timegm(strptime('2011-03-09, 02:00 UTC', f)),
      'end_time'  :timegm(strptime('2011-10-29, 02:00 UTC', f))
    },
    # LCLS run 5: until May 31, 2012
    'CXI 5.1': {
      'address':'CxiDs1-0|Cspad-0',
      'start_time':timegm(strptime('2011-10-29, 02:00 UTC', f)),
      'end_time'  :timegm(strptime('2012-05-31, 02:00 UTC', f))
    },
    # LCLS run 6: until Dec 31, 2012
    'CXI 6.1': {
      'address':'CxiDs1-0|Cspad-0',
      'start_time':timegm(strptime('2012-05-31, 02:00 UTC', f)),
      'end_time'  :timegm(strptime('2013-01-01, 01:00 UTC', f))
    },
    # LCLS run 7: until Jul 31, 2013
    'CXI 7.1': {
      'address':'CxiDs1-0|Cspad-0',
      'start_time':timegm(strptime('2013-01-01, 01:00 UTC', f)),
      'end_time'  :timegm(strptime('2013-08-01, 02:00 UTC', f))
    },
    'CXI 7.d': {
      'address':'CxiDsd-0|Cspad-0',
      'start_time':timegm(strptime('2013-01-01, 01:00 UTC', f)),
      'end_time'  :timegm(strptime('2013-08-01, 02:00 UTC', f))
    },
    'XPP 7.1': {
      'address':'XppGon-0|Cspad-0',
      'start_time':timegm(strptime('2013-01-01, 01:00 UTC', f)),
      'end_time'  :timegm(strptime('2013-08-01, 02:00 UTC', f))
    },
    'XPP 7.marccd': {
      'address':'XppGon-0|marccd-0',
      'start_time':timegm(strptime('2013-01-01, 01:00 UTC', f)),
      'end_time'  :timegm(strptime('2013-08-01, 02:00 UTC', f))
    },
    # LCLS run 8: until Mar 31, 2014
    # CXI detector rebuilt in Jan 2014, before that use CXI 8.1 metrology.
    'CXI 8.1': {
      'address':'CxiDs1-0|Cspad-0',
      'start_time':timegm(strptime('2013-08-01, 02:00 UTC', f)),
      'end_time'  :timegm(strptime('2014-01-15, 02:00 UTC', f))
    },
    'CXI 8.2': {
      'address':'CxiDs1-0|Cspad-0',
      'start_time':timegm(strptime('2014-01-15, 02:00 UTC', f)),
      'end_time'  :timegm(strptime('2014-03-31, 02:00 UTC', f))
    },
    'CXI 8.d': {
      'address':'CxiDsd-0|Cspad-0',
      'start_time':timegm(strptime('2013-08-01, 02:00 UTC', f)),
      'end_time'  :timegm(strptime('2014-03-31, 02:00 UTC', f))
    },
    'XPP 8.1': {
      'address':'XppGon-0|Cspad-0',
      'start_time':timegm(strptime('2013-08-01, 02:00 UTC', f)),
      'end_time'  :timegm(strptime('2014-03-31, 02:00 UTC', f))
    },
    'XPP 8.marccd': {
      'address':'XppGon-0|marccd-0',
      'start_time':timegm(strptime('2013-08-01, 02:00 UTC', f)),
      'end_time'  :timegm(strptime('2014-03-31, 02:00 UTC', f))
    }
  }
_detector_format_version_dict = _get_detector_format_version_dict()

def detector_format_version(address, time):
  """The detector_format_version() function returns a format version
  string appropriate for the detector whose address is given by @p
  address at the time @p time.

  @param address Full data source address of the DAQ device
  @param time    Time of the event, in number of seconds since
                 midnight, 1 January 1970 UTC (Unix time)
  @return        Format version string
  """

  from calendar import timegm
  from time import strptime

  if address is None:
    return None # time can be None if any time is valid for the format version

  ret = None
  for format_name, format in _detector_format_version_dict.iteritems():
    if address == format['address'] and \
       (format['start_time'] is None or time >  format['start_time']) and \
       (format['end_time']   is None or time <= format['end_time']):
      assert ret is None
      ret = format_name

  return ret

def address_and_timestamp_from_detector_format_version(format_name):
  """Reverse of detector_format_version()

  @param format_name detector format version requested
  @return tuple of address and end_time of this format version
  """

  if _detector_format_version_dict.has_key(format_name):
    return _detector_format_version_dict[format_name]['address'], \
           _detector_format_version_dict[format_name]['end_time']
  else:
    return None

