#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <cctbx/eltbx/caasf.h>
#include <scitbx/boost_python/iterator_wrappers.h>

namespace cctbx { namespace eltbx { namespace caasf { namespace boost_python {

namespace {

  struct custom_wrappers : boost::python::pickle_suite
  {
    typedef custom w_t;

    static
    boost::python::tuple
    getinitargs(w_t const& w)
    {
      return boost::python::make_tuple(w.a(), w.b(), w.c());
    }

    static void
    wrap()
    {
      using namespace boost::python;
      typedef return_value_policy<copy_const_reference> ccr;
      class_<w_t>("custom", no_init)
        .def(init<float>())
        .def(init<af::small<float, w_t::max_n_ab> const&,
                  af::small<float, w_t::max_n_ab> const&,
                  float>())
        .def("n_ab", &w_t::n_ab)
        .def("a", (af::small<float, w_t::max_n_ab> const&(w_t::*)()const)
          &w_t::a, ccr())
        .def("b", (af::small<float, w_t::max_n_ab> const&(w_t::*)()const)
          &w_t::b, ccr())
        .def("c", &w_t::c)
        .def("at_stol_sq", &w_t::at_stol_sq)
        .def("at_stol", &w_t::at_stol)
        .def("at_d_star_sq", &w_t::at_d_star_sq)
        .def_pickle(custom_wrappers())
      ;
    }
  };

  template <std::size_t N>
  struct base_wrappers
  {
    typedef base<N> w_t;

    static void
    wrap(const char* python_name)
    {
      using namespace boost::python;
      class_<w_t>(python_name, no_init)
        .def("table", &w_t::table)
        .def("label", &w_t::label)
        .def("a", (af::small<float, custom::max_n_ab>(w_t::*)()const)
          &w_t::a)
        .def("b", (af::small<float, custom::max_n_ab>(w_t::*)()const)
          &w_t::b)
        .def("c", &w_t::c)
        .def("at_stol_sq", &w_t::at_stol_sq)
        .def("at_stol", &w_t::at_stol)
        .def("at_d_star_sq", &w_t::at_d_star_sq)
      ;
    }
  };

  struct it1992_wrappers
  {
    typedef it1992 w_t;

    static void
    wrap()
    {
      using namespace boost::python;
      base_wrappers<4>::wrap("base_4");
      class_<w_t, bases<base<4> > >("it1992", no_init)
        .def(init<std::string const&, optional<bool> >())
      ;
    }
  };

  struct wk1995_wrappers
  {
    typedef wk1995 w_t;

    static void
    wrap()
    {
      using namespace boost::python;
      base_wrappers<5>::wrap("base_5");
      class_<w_t, bases<base<5> > >("wk1995", no_init)
        .def(init<std::string const&, optional<bool> >())
      ;
    }
  };

  void init_module()
  {
    using namespace boost::python;

    custom_wrappers::wrap();

    it1992_wrappers::wrap();
    scitbx::boost_python::iterator_wrappers<
      it1992, it1992_iterator>::wrap("it1992_iterator");

    wk1995_wrappers::wrap();
    scitbx::boost_python::iterator_wrappers<
      wk1995, wk1995_iterator>::wrap("wk1995_iterator");
  }

} // namespace <anonymous>
}}}} // namespace cctbx::eltbx::caasf::boost_python

BOOST_PYTHON_MODULE(caasf_ext)
{
  cctbx::eltbx::caasf::boost_python::init_module();
}
