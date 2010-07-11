#include <smtbx/refinement/constraints/geometrical_hydrogens.h>
#include <scitbx/sparse/io.h>

namespace smtbx { namespace refinement { namespace constraints {

  //*** CH3, NH2, OH ***

  std::size_t terminal_tetrahedral_xhn_sites::size() const {
    return 3*x_h.size();
  }

  void
  terminal_tetrahedral_xhn_sites
  ::linearise(uctbx::unit_cell const &unit_cell,
              sparse_matrix_type *jacobian_transpose)
  {
    using namespace constants;
    site_parameter *pivot = (site_parameter *)argument(0),
                   *pivot_neighbour = (site_parameter *)argument(1);
    independent_scalar_parameter
      *azimuth = (independent_scalar_parameter *)argument(2),
      *length  = (independent_scalar_parameter *)argument(3);

    // Local frame
    cart_t x_p = unit_cell.orthogonalize(pivot->value),
           x_pn = unit_cell.orthogonalize(pivot_neighbour->value);
    af::tiny<cart_t, 3>
    e = scitbx::math::orthonormal_basis(x_p - x_pn, e_zero_azimuth);

    double phi = azimuth->value;
    double l = length->value;
    double cos_phi = std::cos(phi), sin_phi = std::sin(phi);

    // Loop over the Hydrogen atoms
    for (int k=0; k < x_h.size(); ++k) {

      // Cosine and Sine of the azimutal angle of the k-th Hydrogen
      /* Mathematica:
       Table[TrigExpand[Cos[\[Phi] + n Pi/3]], {n, {2, 4}}]
       Table[TrigExpand[Sin[\[Phi] + n Pi/3]], {n, {2, 4}}]
       */
      double c, s;
      switch (k) {
        case 0:
          // 1st Hydrogen: azimuthal angle = phi
          c = cos_phi;
          s = sin_phi;
          break;
        case 1:
          // 2nd Hydrogen: azimuthal angle = phi + 2pi/3
          c = -0.5        *cos_phi - half_sqrt_3*sin_phi;
          s =  half_sqrt_3*cos_phi -         0.5*sin_phi;
          break;
        case 2:
          // 3rd Hydrogen: azimuthal angle = phi + 4pi/3
          c = -0.5        *cos_phi + half_sqrt_3*sin_phi;
          s = -half_sqrt_3*cos_phi -         0.5*sin_phi;
        default:
          break;
      }

      // Site of k-th Hydrogen
      cart_t u = sin_tetrahedral_angle*(c*e[1] + s*e[2]) + e[0]/3.;
      x_h[k] = x_p + l*u;

      // Derivatives
      if (!jacobian_transpose) continue;
      sparse_matrix_type &jt = *jacobian_transpose;
      std::size_t const j_h = index() + 3*k;

      // Riding
      for (int i=0; i<3; ++i) {
        jt.col(j_h + i) = jt.col(pivot->index() + i);
      }

      /** We take advantage of the fact that azimuth and length are
          independent variables. So jt.col(azimuth->index()) is either
          zero or is a column of the identity matrix.
       */

      // Rotation
      if (azimuth->is_variable()) {
        cart_t grad_c = l*sin_tetrahedral_angle*(-s*e[1] + c*e[2]);
        frac_t grad_f = unit_cell.fractionalize(grad_c);
        for (int i=0; i<3; ++i) jt(azimuth->index(), j_h + i) = grad_f[i];
      }

      // Bond stretching
      if (length->is_variable()) {
        frac_t grad_f = unit_cell.fractionalize(u);
        for (int i=0; i<3; ++i) jt(length->index(), j_h + i) = grad_f[i];
      }
    }
  }

  void
  terminal_tetrahedral_xhn_sites::store(uctbx::unit_cell const &unit_cell) const
  {
    for (int i=0; i<hydrogen.size(); ++i) {
      hydrogen[i]->site = unit_cell.fractionalize(x_h[i]);
    }
  }



}}}
