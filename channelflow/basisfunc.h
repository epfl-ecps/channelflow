/**
 * a Complex-Vector-valued Spectral expansion class.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_BASISFUNC_H
#define CHANNELFLOW_BASISFUNC_H
#include <vector>
#include "cfbasics/cfarray.h"
#include "cfbasics/mathdefs.h"
#include "channelflow/chebyshev.h"

namespace chflow {

class FieldSymmetry;

class BasisFunc {
   public:
    BasisFunc();
    BasisFunc(const std::string& filebase);
    BasisFunc(int Nd, int Ny, int kx, int kz, Real Lx, Real Lz, Real a, Real b, fieldstate s = Spectral);
    BasisFunc(int Ny, const BasisFunc& f);

    BasisFunc(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, const ComplexChebyCoeff& w, int kx, int kz,
              Real Lx, Real Lz);

    void save(const std::string& filebase, fieldstate s = Physical) const;
    void binaryDump(std::ostream& os) const;
    void binaryLoad(std::istream& is);

    void randomize(Real magn, Real decay, BC aBC, BC bBC);
    void interpolate(const BasisFunc& f);
    void reflect(const BasisFunc& f);

    inline int Nd() const;
    inline int Ny() const;
    inline int kx() const;
    inline int kz() const;
    inline Real Lx() const;
    inline Real Lz() const;
    inline Real a() const;
    inline Real b() const;
    inline fieldstate state() const;

    void reconfig(const BasisFunc& f);  // set to same geom params, zero value
    void resize(int Ny, int Nd);
    void setBounds(Real Lx, Real Lz, Real a, Real b);
    void setkxkz(int kx, int kz);
    void setState(fieldstate s);

    void setToZero();
    void conjugate();
    void fill(const BasisFunc& f);

    void chebyfft();
    void ichebyfft();
    void makeSpectral();
    void makePhysical();
    void makeState(fieldstate s);

    void chebyfft(const ChebyTransform& t);
    void ichebyfft(const ChebyTransform& t);
    void makeSpectral(const ChebyTransform& t);
    void makePhysical(const ChebyTransform& t);
    void makeState(fieldstate s, const ChebyTransform& t);

    const ComplexChebyCoeff& u() const;
    const ComplexChebyCoeff& v() const;
    const ComplexChebyCoeff& w() const;
    const ComplexChebyCoeff& operator[](int i) const;  // ith component
    ComplexChebyCoeff& u();
    ComplexChebyCoeff& v();
    ComplexChebyCoeff& w();
    ComplexChebyCoeff& operator[](int i);  // ith component

    bool geomCongruent(const BasisFunc& f) const;
    bool congruent(const BasisFunc& f) const;      // geomCongruent & kx,kz equality
    bool interoperable(const BasisFunc& f) const;  // congruent && same state

    BasisFunc& operator*=(const FieldSymmetry& s);
    BasisFunc& operator*=(const BasisFunc& g);
    BasisFunc& operator+=(const BasisFunc& g);
    BasisFunc& operator-=(const BasisFunc& g);
    BasisFunc& operator*=(Real c);
    BasisFunc& operator*=(Complex c);  // gcc-2.95 can't handle thi

    Real bcNorm(BC aBC, BC bBC) const;

   protected:
    int Nd_;
    int Ny_;
    int kx_;
    int kz_;
    Real Lx_;
    Real Lz_;
    Real a_;
    Real b_;
    fieldstate state_;
    cfarray<ComplexChebyCoeff> u_;
};

class BasisFlags {
   public:
    BasisFlags(BC aBC = Diri, BC bBC = Diri, bool zerodiv = true, bool orthonorm = true);
    BC aBC;
    BC bBC;
    bool zerodivergence;
    bool orthonormalize;
};

// Construct a basis for (kx,kz) Fourier subspace of 3d velocity fields on
// the given domain, up to order-Ny polynomials in y.
std::vector<BasisFunc> complexBasisKxKz(int Ny, int kx, int kz, Real Lx, Real Lz, Real a, Real b,
                                        const BasisFlags& flags);

// Construct a basis for -kxmax<=kx<=kxmax, 0<=kz<=kzmax, order Ny polynomials
std::vector<BasisFunc> complexBasis(int Ny, int kxmax, int kzmax, Real Lx, Real Lz, Real a, Real b,
                                    const BasisFlags& flags);

void orthonormalize(std::vector<BasisFunc>& f);

void checkBasis(const std::vector<BasisFunc>& e, const BasisFlags& flags, bool orthogcheck = false);

bool operator==(const BasisFunc& f, const BasisFunc& g);
bool operator!=(const BasisFunc& f, const BasisFunc& g);

BasisFunc conjugate(const BasisFunc& f);

// L2Norm2(f)  == Int ||f||^2     dx dy dz (/(Lx Ly Lz)    if normalize==true)
// bcNorm2(f)  == Int ||f||^2     dx dz at y=a,b (/(Lx Lz) if normalize==true)
// divNorm2(f) == Int ||div f||^2 dx dy dz (/(Lx Ly Lz)    if normalize==true)

// innerProduct(f,g) == Int f g*  dx dy dz (/(Lx Ly Lz)    if normalize==true)

// L2Norm(f)    == sqrt(L2Norm2(f))
// L2Dist(f,g)  == sqrt(L2Dist2(f,g))
// L2Dist2(f,g) == L2Norm2(f-g)
// etc.

// These are not so simple to compute for individual Fourier modes.
// Real L1Norm(const BasisFunc& f, bool normalize=true);
// Real L1Dist(const BasisFunc& f, const BasisFunc& g, bool normalize=true);
// Real LinfNorm(const BasisFunc& f);
// Real LinfDist(const BasisFunc& f, const BasisFunc& g);

Real L2Norm(const BasisFunc& f, bool normalize = true);
Real L2Norm2(const BasisFunc& f, bool normalize = true);
Real L2Dist(const BasisFunc& f, const BasisFunc& g, bool normalize = true);
Real L2Dist2(const BasisFunc& f, const BasisFunc& g, bool normalize = true);
Complex L2InnerProduct(const BasisFunc& f0, const BasisFunc& f1, bool normalize = true);

Real chebyNorm(const BasisFunc& f, bool normalize = true);
Real chebyNorm2(const BasisFunc& f, bool normalize = true);
Real chebyDist(const BasisFunc& f, const BasisFunc& g, bool normalize = true);
Real chebyDist2(const BasisFunc& f, const BasisFunc& g, bool normalize = true);
Complex chebyInnerProduct(const BasisFunc& f0, const BasisFunc& f1, bool normalize = true);

Real norm(const BasisFunc& f, NormType n, bool normalize = true);
Real norm2(const BasisFunc& f, NormType n, bool normalize = true);
Real dist(const BasisFunc& f, const BasisFunc& g, NormType n, bool nrmlz = true);
Real dist2(const BasisFunc& f, const BasisFunc& g, NormType n, bool nrmlz = true);
Complex innerProduct(const BasisFunc& f0, const BasisFunc& f1, NormType n, bool normalize = true);

Real bcNorm(const BasisFunc& f, bool normalize = true);
Real bcNorm2(const BasisFunc& f, bool normalize = true);
Real bcDist(const BasisFunc& f, const BasisFunc& g, bool normalize = true);
Real bcDist2(const BasisFunc& f, const BasisFunc& g, bool normalize = true);

Real divNorm(const BasisFunc& f, bool normalize = true);
Real divNorm2(const BasisFunc& f, bool normalize = true);
Real divDist(const BasisFunc& f, const BasisFunc& g, bool normalize = true);
Real divDist2(const BasisFunc& f, const BasisFunc& g, bool normalize = true);

BasisFunc xdiff(const BasisFunc& f);
BasisFunc ydiff(const BasisFunc& f);
BasisFunc zdiff(const BasisFunc& f);
BasisFunc grad(const BasisFunc& f);  // rtn[Nd*i + j] = df_i/dx_j
BasisFunc curl(const BasisFunc& f);
BasisFunc lapl(const BasisFunc& f);
BasisFunc div(const BasisFunc& f);
BasisFunc dot(const BasisFunc& f, const BasisFunc& g);
BasisFunc cross(const BasisFunc& f, const BasisFunc& g);
BasisFunc dotgrad(const BasisFunc& f_n, const BasisFunc& f_p);

void dotgrad(const BasisFunc& f_n, const BasisFunc& grad_f_pu, const BasisFunc& grad_f_pv, const BasisFunc& grad_f_pw,
             ComplexChebyCoeff& tmp, BasisFunc& rtn);

void xdiff(const BasisFunc& f, BasisFunc& fx);
void ydiff(const BasisFunc& f, BasisFunc& fy);
void zdiff(const BasisFunc& f, BasisFunc& fz);
void grad(const BasisFunc& f, BasisFunc& grad_f);
void lapl(const BasisFunc& f, BasisFunc& lapl_f);
void curl(const BasisFunc& f, BasisFunc& curl_f);
void div(const BasisFunc& f, BasisFunc& div_f);
void dot(const BasisFunc& f, const BasisFunc& g, BasisFunc& f_dot_g);
void cross(const BasisFunc& f, const BasisFunc& g, BasisFunc& f_cross_g);
void dotgrad(const BasisFunc& f, const BasisFunc& g, BasisFunc& f_dotgrad_g);

void ubcFix(ChebyCoeff& u, BC aBC, BC bBC);
void vbcFix(ChebyCoeff& v, BC aBC, BC bBC);
void ubcFix(ComplexChebyCoeff& u, BC aBC, BC bBC);
void vbcFix(ComplexChebyCoeff& v, BC aBC, BC bBC);

inline int BasisFunc::Nd() const { return Nd_; }
inline int BasisFunc::Ny() const { return Ny_; }
inline int BasisFunc::kx() const { return kx_; }
inline int BasisFunc::kz() const { return kz_; }
inline Real BasisFunc::Lx() const { return Lx_; }
inline Real BasisFunc::Lz() const { return Lz_; }
inline Real BasisFunc::a() const { return a_; }
inline Real BasisFunc::b() const { return b_; }
inline fieldstate BasisFunc::state() const { return state_; }

// helper function for emulating 2-tensors with 9d fields
inline int i3j(int i, int j) { return 3 * i + j; }

}  // namespace chflow

#endif
