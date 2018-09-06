/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_REALPROFILE_NG
#define CHANNELFLOW_REALPROFILE_NG

#include <vector>
#include "channelflow/chebyshev.h"
#include "channelflow/symmetry.h"

namespace chflow {

/**
 * Represents a vector field on the xz-periodic domain [0, Lx] x [a,b] x [0,Lz]
 * of the form :       Phi_u(y) g_jx{alpha x} g_-jz{gamma z} xhat +
 *                             Phi_v(y) g_-jx{alpha x} g_-jz{gamma z} yhat +
 *                              Phi_w(y) g_-jx{alpha x} g_jz{gamma z} zhat
 *
 * Where               [ cos(j * x) for  j >= 0
 *            g_j(x) = {
 *                         [ sin(-j * x) for J < 0
 */
class RealProfileNG {
   public:
    RealProfileNG(const int jx, const int jz, const int Nd, const int Ny, const Real Lx, const Real Lz, const Real a,
                  const Real b, const fieldstate state = Spectral);

    RealProfileNG(const std::vector<ChebyCoeff> u, const int jx, const int jz, const Real Lx, const Real Lz);

    RealProfileNG();

    RealProfileNG(const RealProfileNG&);

    inline int jx() const;
    inline int jz() const;
    inline int Nd() const;
    inline int Ny() const;
    inline Real Lx() const;
    inline Real Lz() const;
    inline Real a() const;
    inline Real b() const;
    inline fieldstate state() const;
    inline void setJx(int jx);
    inline void setJz(int jz);
    RealProfileNG& operator=(const RealProfileNG&);
    RealProfileNG& operator*=(const Real c);
    RealProfileNG& operator+=(const RealProfileNG& e);
    RealProfileNG& operator-=(const RealProfileNG& e);
    RealProfileNG& operator*=(const FieldSymmetry& s);
    inline const ChebyCoeff& operator[](int i) const;
    inline ChebyCoeff& operator[](int i);

    // Can be added/subtracted to e (true if congruent and on same geometry)
    bool compatible(const RealProfileNG& e) const;
    // True if on some geometry and in same state
    bool congruent(const RealProfileNG& e) const;

    void makeSpectral();                 // if Physical, transform to Spectral
    void makePhysical();                 // if Spectral, transform to Physical
    void makeState(const fieldstate s);  // if state != s, transform to state s

    void makeSpectral(const ChebyTransform& t);
    void makePhysical(const ChebyTransform& t);
    void makeState(const fieldstate s, const ChebyTransform& t);

    // When converting to a FlowField
    // Gives the appropriate normalization factor for the (+kx,kz) fourier mode
    Complex normalization_p(const int d) const;

    // Gives the appropriate normalization factor for the (-kx,kz) fourier mode
    Complex normalization_m(const int d) const;

   private:
    fieldstate state_;
    int jx_;
    int jz_;
    int Nd_;
    int Ny_;
    Real Lx_;
    Real Lz_;
    Real a_;
    Real b_;

   public:
    std::vector<ChebyCoeff> u_;
};

inline const ChebyCoeff& RealProfileNG::operator[](int i) const {
    assert(i >= 0 && i < Nd());
    return u_[i];
}
inline ChebyCoeff& RealProfileNG::operator[](int i) {
    assert(i >= 0 && i < Nd());
    return u_[i];
}

inline int RealProfileNG::jx() const { return jx_; }
inline int RealProfileNG::jz() const { return jz_; }
inline int RealProfileNG::Nd() const { return Nd_; }
inline int RealProfileNG::Ny() const { return Ny_; }
inline Real RealProfileNG::Lx() const { return Lx_; }
inline Real RealProfileNG::Lz() const { return Lz_; }
inline Real RealProfileNG::a() const { return a_; }
inline Real RealProfileNG::b() const { return b_; }
inline fieldstate RealProfileNG::state() const { return state_; }
inline void RealProfileNG::setJx(int jx) { jx_ = jx; }
inline void RealProfileNG::setJz(int jz) { jz_ = jz; }

Real L2InnerProduct(const RealProfileNG& e1, const RealProfileNG& e2, const bool normalize = true);
Real L2Norm2(const RealProfileNG& e, bool normalize = true);
inline Real L2Norm(const RealProfileNG& e, bool normalize = true) { return sqrt(L2Norm2(e, normalize)); }

std::vector<RealProfileNG> realBasisNG(const int Ny, const int kxmax, const int kzmax, const Real Lx, const Real Lz,
                                       const Real a, const Real b);

void orthonormalize(std::vector<RealProfileNG>& basis);

// Remove all elements from basis which are not symmetric under all members of s, to a given tolerence
void selectSymmetries(std::vector<RealProfileNG>& basis, const std::vector<FieldSymmetry>& s,
                      const Real tolerance = 1e-13);
}  // namespace chflow

#endif
