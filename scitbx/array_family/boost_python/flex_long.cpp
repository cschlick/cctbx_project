/* Copyright (c) 2001-2002 The Regents of the University of California
   through E.O. Lawrence Berkeley National Laboratory, subject to
   approval by the U.S. Department of Energy.
   See files COPYRIGHT.txt and LICENSE.txt for further details.

   Revision history:
     2002 Aug: Created (R.W. Grosse-Kunstleve)
 */

#include <scitbx/array_family/boost_python/flex_wrapper.h>

namespace scitbx { namespace af { namespace boost_python {

  void wrap_flex_long_2(flex_wrapper<long>::class_f_t class_object);

  void wrap_flex_long()
  {
    wrap_flex_long_2(
      flex_wrapper<long>::signed_integer("long", boost::python::scope()));
  }

}}} // namespace scitbx::af::boost_python
