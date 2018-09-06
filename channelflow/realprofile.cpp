/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/realprofile.h"
#include <fstream>
#include <iomanip>
#include "cfbasics/mathdefs.h"

using namespace std;

namespace chflow {

RealProfile::RealProfile() : psi(), sign_(Plus) {}

RealProfile::RealProfile(int Ny, const RealProfile& f) : psi(Ny, f.psi), sign_(f.sign_) {}

RealProfile::RealProfile(const BasisFunc& phi, Sign sign) : psi(phi), sign_(sign) {
    // This normalizes f = (psi + psi*) so that
    // if L2Norm(psi) == 1, then L2Norm(f) == 1.
    if (psi.kx() == 0 && psi.kz() == 0)
        psi *= 0.5;
    else
        psi *= sqrt(0.5);
}

RealProfile::RealProfile(int Nd, int Ny, int kx, int kz, Real Lx, Real Lz, Real a, Real b, Sign sig, fieldstate stat)
    : psi(Nd, Ny, kx, kz, Lx, Lz, a, b, stat), sign_(sig) {}

RealProfile::RealProfile(const ComplexChebyCoeff& u, const ComplexChebyCoeff& v, const ComplexChebyCoeff& w, int kx,
                         int kz, Real Lx, Real Lz, Sign s)
    : psi(u, v, w, kx, kz, Lx, Lz), sign_(s) {}

void RealProfile::save(const string& filebase, fieldstate savestate) const {
    fieldstate origstate = psi.state();
    ((RealProfile&)*this).makeState(savestate);

    string filename(filebase);
    filename += string(".bf");
    ofstream os(filename.c_str());
    os << scientific << setprecision(REAL_DIGITS);
    char sp = ' ';
    os << '%' << sp << psi.Nd() << sp << psi.Ny() << sp << psi.kx() << sp << psi.kz() << sp << psi.Lx() << sp
       << psi.Lz() << sp << psi.a() << sp << psi.b() << sp << psi.state() << sp << sign_ << '\n';

    int Ny = psi.Ny();
    int Nd = psi.Nd();

    // RealProfile f == (phi + phi^*)    == Re phi, sign == Plus
    //             f == (phi - phi^*)/i  == Im phi, sign == Minus
    for (int ny = 0; ny < Ny; ++ny) {
        for (int n = 0; n < Nd; ++n)
            os << psi[n].re[ny] << sp << psi[n].im[ny] << sp;
        os << '\n';
    }
    os.close();

    ((RealProfile&)*this).makeState(origstate);
}

void RealProfile::binaryDump(ostream& os) const {
    write(os, int(sign_));
    psi.binaryDump(os);
}

void RealProfile::binaryLoad(istream& is) {
    if (!is.good())
        cferror("RealProfile::binaryLoad(istream) : input error");
    int s;
    read(is, s);
    switch (s) {
        case -1:
            sign_ = Minus;
            break;
        case 1:
            sign_ = Plus;
            break;
        default:
            cferror("RealProfile::binaryLoad(istream) : io error on sign member");
    }
    psi.binaryLoad(is);
}

/******************************************
void RealProfile::consistencyCheck() const {
  if (psi.kx()==0 && psi.kz()==0 && sign
_ != Zero)
    cferror("RealProfile::consistencyCheck() : kx==0 && kz==0 && Sign != Zero");
  else if (psi.kx()!=0 && psi.kz()!=0 && sign_ == Zero)
    cferror("RealProfile::consistencyCheck() : kx!=0 && kz!=0 && Sign == Zero");
}
********************************************/

// Canonical form of RealProfile is
// f == psi +/- (psi*)  with psi.kz>0   or  (psi.kz == 0 and psi.kx > 0)

// If those conditions aren't met, conjugate psi and swap around signs
// so that psi +/- (psi*) has same value but is in canonical form

// If psi is not canonical,
// For Plus, f == psi + psi*.    Let psi' == psi*
// Then      f == psi' + (psi'*)

// For Minus f == -i*(psi - psi*).    Let psi' = -psi*
// Then      f == -i*(psi' - (psi'*))

#include <iomanip>

void RealProfile::canonicalize() {
    // cout << "canonicalizing" << endl;
    // cout << "start kx,kz == " << psi.kx() << ", " << psi.kz() << endl;

    if (psi.kz() < 0 || (psi.kz() == 0 && psi.kx() < 0)) {
        // cout << "conjugating" << endl;
        psi.conjugate();
        if (sign_ == Minus) {
            // cout << "flipping sign" << endl;
            psi *= Complex(0, -1);
        }
    }
    // cout << "end   kx,kz == " << psi.kx() << ", " << psi.kz() << endl;
}

void RealProfile::randomize(Real magn, Real decay, BC aBC, BC bBC) {
    if (psi.kx() == 0 && psi.kz() == 0)
        sign_ = Plus;
    else
        sign_ = (rand() % 2 == 0) ? Minus : Plus;
    psi.randomize(magn, decay, aBC, bBC);
}

void RealProfile::interpolate(const RealProfile& f) {
    sign_ = f.sign_;
    psi.interpolate(f.psi);
}

void RealProfile::reflect(const RealProfile& f) { psi.reflect(f.psi); }

void RealProfile::resize(int Ny, int Nd) { psi.resize(Ny, Nd); }

void RealProfile::reconfig(const RealProfile& f) {
    psi.reconfig(f.psi);
    sign_ = f.sign_;
}

void RealProfile::setBounds(Real Lx, Real Lz, Real a, Real b) { psi.setBounds(Lx, Lz, a, b); }
void RealProfile::setkxkzSign(int kx, int kz, Sign s) {
    psi.setkxkz(kx, kz);
    sign_ = s;
}

void RealProfile::setState(fieldstate s) { psi.setState(s); }

void RealProfile::setSign(Sign s) { sign_ = s; }

void RealProfile::setToZero() { psi.setToZero(); }

void RealProfile::fill(const RealProfile& f) { psi.fill(f.psi); }

bool RealProfile::geomCongruent(const RealProfile& f) const { return psi.geomCongruent(f.psi); }

bool RealProfile::congruent(const RealProfile& f) const { return (psi.congruent(f.psi) && sign_ == f.sign_); }

bool RealProfile::interoperable(const RealProfile& f) const { return (psi.interoperable(f.psi) && sign_ == f.sign_); }

void RealProfile::chebyfft() { psi.chebyfft(); }

void RealProfile::ichebyfft() { psi.ichebyfft(); }

void RealProfile::makeSpectral() { psi.makeSpectral(); }

void RealProfile::makePhysical() { psi.makePhysical(); }

void RealProfile::makeState(fieldstate s) { psi.makeState(s); }

void RealProfile::chebyfft(const ChebyTransform& t) { psi.chebyfft(t); }

void RealProfile::ichebyfft(const ChebyTransform& t) { psi.ichebyfft(t); }

void RealProfile::makeSpectral(const ChebyTransform& t) { psi.makeSpectral(t); }

void RealProfile::makePhysical(const ChebyTransform& t) { psi.makePhysical(t); }

void RealProfile::makeState(fieldstate s, const ChebyTransform& t) { psi.makeState(s, t); }

/*******************************************************
ChebyCoeff& RealProfile::u() const {return (*this)[0];}
ChebyCoeff& RealProfile::v() const {return (*this)[1];}
ChebyCoeff& RealProfile::w() const {return (*this)[2];}
*******************************************************/

RealProfile RealProfile::operator[](int i) const {
    RealProfile rtn(1, psi.Ny(), psi.kx(), psi.kz(), psi.Lx(), psi.Lz(), psi.a(), psi.b(), sign_, psi.state());
    int Ny = psi.Ny();
    for (int ny = 0; ny < Ny; ++ny)
        rtn.psi[0].set(ny, psi[i][ny]);
    return rtn;
}

/***************************************************************
RealProfile& RealProfile::operator *= (const RealProfile& f) {
  assert(geomCongruent(psi) && psi.state() == Physical && f.psi.state() == Physical);
  for (int n=0; n<psi.Nd(); ++n)
    psi[n] *= f.psi[n];
  return *this;
}
**************************************************************/

RealProfile& RealProfile::operator+=(const RealProfile& f) {
    assert(interoperable(f));
    for (int n = 0; n < psi.Nd(); ++n)
        psi[n] += f.psi[n];
    return *this;
}

RealProfile& RealProfile::operator-=(const RealProfile& f) {
    assert(interoperable(f));
    for (int n = 0; n < psi.Nd(); ++n)
        psi[n] -= f.psi[n];
    return *this;
}

RealProfile& RealProfile::operator*=(Real c) {
    for (int n = 0; n < psi.Nd(); ++n)
        psi[n] *= c;
    return *this;
}

RealProfile& RealProfile::operator*=(const FieldSymmetry& s) {
    psi *= s;
    canonicalize();
    return *this;
}

bool operator!=(const RealProfile& f, const RealProfile& g) { return (f == g) ? false : true; }

bool operator==(const RealProfile& f, const RealProfile& g) {
    if (!f.congruent(g))
        return false;

    for (int n = 0; n < f.Nd(); ++n)
        if (f[n] != g[n])
            return false;

    return true;
}

// ==================================================================
// L2 norms
Real L2Norm(const RealProfile& f, bool normalize) {
    // INEFFICIENT
    return sqrt(L2Norm2(f, normalize));
}

Real L2Norm2(const RealProfile& f, bool normalize) {
    // INEFFICIENT
    Real rtn = Re(L2Norm2(f.psi, normalize));
    if (f.psi.kx() == 0 && f.psi.kz() == 0)
        rtn *= 4;  // ||Psi + Psi*||^2
    else
        rtn *= 2;  // ||Psi||^2 + ||Psi*||^2

    return rtn;
}

Real L2Dist2(const RealProfile& f, const RealProfile& g, bool normalize) {
    assert(f.geomCongruent(g));

    // INEFFICIENT
    RealProfile f_g = f;
    f_g -= g;
    return L2Norm2(f_g, normalize);
}

Real L2Dist(const RealProfile& f, const RealProfile& g, bool normalize) { return sqrt(L2Dist2(f, g, normalize)); }

Real L2InnerProduct(const RealProfile& f, const RealProfile& g, bool normalize) {
    assert(f.geomCongruent(g));
    Real ip = 0.0;
    if ((abs(f.kx()) == abs(g.kx())) && (abs(f.kz()) == abs(g.kz()))) {
        const BasisFunc& a = f.psi;
        const BasisFunc& b = g.psi;

        // INEFFICIENT break below inner products into components
        BasisFunc bs = conjugate(b);

        Complex x = L2InnerProduct(a, b);
        Complex y = L2InnerProduct(a, bs);

        Sign asign = f.sign();
        Sign bsign = g.sign();

        if (asign == Plus) {
            if (bsign == Plus)

                // f = a+a*
                // g = b+b*
                // (a+a*, b+b*) == (a,b) + (a,b*) + (a*,b) + (a*,b*)
                ip = 2 * Re(x + y);

            else
                // f = a+a*
                // g = (b-b*)/i
                // (a+a*, (b-b*)/i) == -(a+a*, (b-b*))/i
                //                  == [-(a,b) + (a,b*) - (a*,b) + (a*,b*)]/i
                //                  == [-(a,b) + (a*,b*) + (a,b*) - (a*,b)]/i
                //                  == -2 Im(a,b) + 2 Im(a,b*)
                ip = 2 * Im(y - x);
        } else if (asign == Minus) {
            if (bsign == Plus)

                // f = (a-a*)/i
                // g =  b+b*
                // (a-a*, b+b*)/i == [(a,b) + (a,b*)  - (a*,b) - (a*,b*)]/i
                //                == [(a,b) - (a*,b*) + (a,b*) - (a*,b) ]/i
                //                == 2 Im(a,b) + 2 Im(a,b*)
                ip = 2 * Im(x + y);
            else
                // f = (a-a*)/i
                // g = (b-b*)/i
                // ((a-a*)/i, (b-b*)/i) == (a-a*, b-b*)
                //                      == [(a,b) - (a,b*)  - (a*,b) + (a*,b*)]
                //                      == [(a,b) + (a*,b*) - (a,b*) - (a*,b)]
                //                      == 2 Re(a,b) - 2 Re(a,b*)
                ip = 2 * Re(x - y);
        }
        // Double if cross-conjugate terms survive
        if (f.kx() == 0 && f.kz() == 0)
            ip *= 2;
    }
    return ip;
}

/*********************************************************************
// This code is merely textually converted from basisfunc code.
// Have not thought through necessary modifications for realprofile.

// ==================================================================
// cheby norms
Real chebyNorm(const RealProfile& psi, bool normalize) {
  Real rtn2 = 0.0;
  for (int n=0; n<psi.Nd(); ++n)
    rtn2 += chebyNorm2(psi[n], normalize);
  if (!normalize)
    rtn2 *= psi.Lx()*psi.Lz();
  return sqrt(rtn2);
}

Real chebyNorm2(const RealProfile& psi, bool normalize) {
  Real rtn = 0.0;
  for (int n=0; n<psi.Nd(); ++n)
    rtn += chebyNorm2(psi[n], normalize);
  if (!normalize)
    rtn *= psi.Lx()*psi.Lz();
  return rtn;
}

Real chebyDist2(const RealProfile& f, const RealProfile& g, bool normalize) {
  assert(f.geomCongruent(g));
  Real rtn=0.0;
  if (f.kx()==g.kx() && f.kz()==g.kz())
    for (int n=0; n<f.Nd(); ++n)
      rtn += chebyDist2(f[n], g[n], normalize);
  if (!normalize)
    rtn *= f.Lx()*f.Lz();
  return rtn;

}

Real chebyDist(const RealProfile& f, const RealProfile& g, bool normalize) {
    return sqrt(chebyDist2(f,g,normalize));
}

Real chebyInnerProduct(const RealProfile& f, const RealProfile& g,
                          bool normalize) {
  assert(f.geomCongruent(g));
  Real rtn = 0.0;
  if (f.sign() == g.sign()) {
    // INEFFICIENT
    Complex ip = L2InnerProduct(f.psi, g.psi);
    switch (f.sign()) {
    case Minus:
      rtn = sqrt(2)*Im(ip);
      break;
    case Zero:
      rtn = Re(ip);
      break;
    case Plus:
      rtn = sqrt(2)*Re(ip);
      break;
    }
  }
  if (!normalize)
    rtn *= f.Lx() * f.Lz();
  if (!normalize)
    rtn *= f.Lx() * f.Lz();
  return rtn;
}
// ==================================================================
// switchable norms
Real norm(const RealProfile& psi, NormType n, bool normalize) {
  return (n==Uniform) ? L2Norm(psi,normalize) : chebyNorm(psi,normalize);
}

Real norm2(const RealProfile& psi, NormType n, bool normalize) {
  return (n==Uniform) ? L2Norm2(psi,normalize) : chebyNorm2(psi,normalize);
}

Real dist2(const RealProfile& f, const RealProfile& g, NormType n, bool nrmlz) {
  return (n==Uniform) ? L2Dist2(f,g,nrmlz) : chebyDist2(f,g,nrmlz);
}

Real dist(const RealProfile& f, const RealProfile& g,  NormType n, bool nrmlz) {
  return (n==Uniform) ? L2Dist(f,g,nrmlz) : chebyDist(f,g,nrmlz);
}

Real innerProduct(const RealProfile& f, const RealProfile& g, NormType n,
                       bool nrmlz) {
  return (n==Uniform) ? L2InnerProduct(f,g,nrmlz) : chebyInnerProduct(f,g,nrmlz);
}
***************************************************************/

// =================================================================
// boundary and divergence norms

Real divNorm2(const RealProfile& f, bool normalize) {
    assert(f.state() == Spectral);
    BasisFunc divf = div(f.psi);
    Real rtn = 2 * L2Norm2(divf, normalize);  // 1 for |psi|^2 + 1 for |psi*|^2
    if (f.kx() == 0 && f.kz() == 0)
        rtn *= 2;  // 1 for |psi psi*| + 1 for |psi*psi|^2
    return rtn;
}

Real divNorm(const RealProfile& f, bool normalize) { return sqrt(divNorm2(f, normalize)); }

Real divDist2(const RealProfile& f, const RealProfile& g, bool normalize) {
    // INEFFICIENT
    assert(f.state() == Spectral && g.state() == Spectral);
    assert(f.interoperable(g));
    assert(f.Nd() == 3);

    RealProfile f_g = f;
    f_g -= g;

    return divNorm2(f_g);
}

Real divDist(const RealProfile& f, const RealProfile& g, bool normalize) { return sqrt(divDist2(f, g, normalize)); }

Real bcNorm2(const RealProfile& f, bool normalize) {
    Real bc2 = 0.0;

    // Calculate || f.psi|_boundary ||^2
    for (int n = 0; n < f.Nd(); ++n) {
        bc2 += square(f.psi[n].re.eval_a());
        bc2 += square(f.psi[n].re.eval_b());
        bc2 += square(f.psi[n].im.eval_a());
        bc2 += square(f.psi[n].im.eval_b());
    }
    // Double result to add || f.psi*|_boundary ||^2
    bc2 *= 2;

    // Double further if conjugate cross terms survive
    if (f.kx() == 0 && f.kz() == 0)
        bc2 *= 2;

    if (!normalize)
        bc2 *= f.Lx() * f.Lz();

    return bc2;
}

Real bcNorm(const RealProfile& f, bool normalize) { return sqrt(bcNorm2(f, normalize)); }

Real bcDist2(const RealProfile& f, const RealProfile& g, bool normalize) {
    assert(f.geomCongruent(g));
    Real bc2 = 0.0;

    // Calculate || (f.psi - g.psi)|_boundary ||^2
    if (f.kx() == g.kx() && f.kz() == g.kz()) {
        for (int n = 0; n < f.Nd(); ++n) {
            bc2 += square(f.psi[n].re.eval_a() - g.psi[n].re.eval_a());
            bc2 += square(f.psi[n].im.eval_a() - g.psi[n].im.eval_a());
            bc2 += square(f.psi[n].re.eval_b() - g.psi[n].re.eval_b());
            bc2 += square(f.psi[n].im.eval_b() - g.psi[n].im.eval_b());
        }
        // Double result to add || (f.psi* - g.psi*)|_boundary ||^2
        bc2 *= 2;

        // Double further if conjugate cross terms survive
        if (f.kx() == 0 && f.kz() == 0)
            bc2 *= 2;
    }
    if (!normalize)
        bc2 *= f.Lx() * f.Lz();
    return bc2;
}

Real bcNorm(const RealProfile& f, const RealProfile& g, bool normalize) { return sqrt(bcNorm2(f, normalize)); }

RealProfile xdiff(const RealProfile& f) { return RealProfile(xdiff(f.psi), f.sign()); }

RealProfile ydiff(const RealProfile& f) { return RealProfile(ydiff(f.psi), f.sign()); }

RealProfile zdiff(const RealProfile& f) { return RealProfile(zdiff(f.psi), f.sign()); }

RealProfile lapl(const RealProfile& f) { return RealProfile(lapl(f.psi), f.sign()); }

// RealProfile grad(const RealProfile& f, int i) {
// return RealProfile(grad(f.psi,i), f.sign());
//}

RealProfile grad(const RealProfile& f) { return RealProfile(grad(f.psi), f.sign()); }

RealProfile curl(const RealProfile& f) { return RealProfile(curl(f.psi), f.sign()); }

RealProfile div(const RealProfile& f) { return RealProfile(div(f.psi), f.sign()); }

void xdiff(const RealProfile& f, RealProfile& rtn) {
    xdiff(f.psi, rtn.psi);
    rtn.setSign(f.sign());
}
void ydiff(const RealProfile& f, RealProfile& rtn) {
    ydiff(f.psi, rtn.psi);
    rtn.setSign(f.sign());
}
void zdiff(const RealProfile& f, RealProfile& rtn) {
    zdiff(f.psi, rtn.psi);
    rtn.setSign(f.sign());
}

void lapl(const RealProfile& f, RealProfile& rtn) {
    lapl(f.psi, rtn.psi);
    rtn.setSign(f.sign());
}

void grad(const RealProfile& f, RealProfile& rtn) {
    grad(f.psi, rtn.psi);
    rtn.setSign(f.sign());
}

void curl(const RealProfile& f, RealProfile& rtn) {
    curl(f.psi, rtn.psi);
    rtn.setSign(f.sign());
}

void div(const RealProfile& f, RealProfile& divf) {
    div(f.psi, divf.psi);
    divf.setSign(f.sign());
}

// f,g are RealProfiles; a,b are complex BasisFuncs
//
// f == a - a*  Minus
//   or a + a*  Plus
//
// g == b - b*  Minus
//   or b + b*  Plus

// In what follows, terms in brackets are conjugates of non-parenthesized
// terms, added/subtracted implicitly in the rtn value, according to Minus/Plus

// Case Plus Plus
// (a+a*) x (b+b*) == a x b + a x b* + a* x b + a* x b*
//
//            rtn1 == a x b   + [a x b]*
//            rtn2 == a x b*  + [a* x b]

// Case Minus Plus
// (a-a*) x (b+b*) == a x b + a x b* - a* x b - a* x b*
//
//            rtn1 == a x b   - [a x b]*
//            rtn2 == a x b*  - [a* x b]

// Case Plus Minus
// (a+a*) x (b-b*) == a x b - a x b* + a* x b - a* x b*
//
//            rtn1 == a x b   - [a x b]*
//            rtn2 == -a x b* + [a* x b]

// Case Minus Minus
// (a-a*) x (b-b*) == a x b - a x b* - a* x b + a* x b*
//
//            rtn1 == a x b   + [a x b]*
//            rtn2 == -a x b* - [a* x b]

void cross(const RealProfile& f, const RealProfile& g, RealProfile& rtn1, RealProfile& rtn2) {
    assert(f.geomCongruent(g));
    assert(f.Nd() == 3);
    assert(g.Nd() == 3);

    rtn1.resize(f.Ny(), 1);
    rtn2.resize(f.Ny(), 1);

    const BasisFunc& a = f.psi;
    const BasisFunc& b = g.psi;
    const BasisFunc& bs = conjugate(b);

    cross(a, b, rtn1.psi);
    cross(a, bs, rtn2.psi);

    multiplySetSigns(f.sign(), g.sign(), rtn1, rtn2);
}

void dot(const RealProfile& f, const RealProfile& g, RealProfile& rtn1, RealProfile& rtn2) {
    assert(f.Lx() == g.Lx() && f.Lz() == g.Lz() && f.a() == g.a() && f.Ny() == g.Ny());
    assert(f.Nd() == 3);
    assert(g.Nd() == 3 || g.Nd() == 9);

    const BasisFunc& a = f.psi;
    const BasisFunc& b = g.psi;
    const BasisFunc bs = conjugate(b);

    dot(a, b, rtn1.psi);
    dot(a, bs, rtn2.psi);

    multiplySetSigns(f.sign(), g.sign(), rtn1, rtn2);
}

void dotgrad(const RealProfile& f, const RealProfile& g, RealProfile& rtn1, RealProfile& rtn2) {
    RealProfile gradg;
    grad(g, gradg);
    dot(f, gradg, rtn1, rtn2);
}

// Coming in,
// rtn1.psi == a x b,
// rtn2.psi == a x b*,
// Coming out,
// rtn1 and rtn2 have signs and Signs set according to signs of a+-a*, b+-b*
void multiplySetSigns(Sign asign, Sign bsign, RealProfile& rtn1, RealProfile& rtn2) {
    // Case a,b == Plus,Plus
    // f x g == (a+a*) x (b+b*)
    //       == a x b + a x b* + a* x b + a* x b*
    //  rtn1 == a x b   + [a x b]*
    //  rtn2 == a x b*  + [a* x b]
    if (asign == Plus && bsign == Plus) {
        rtn1.setSign(Plus);
        rtn2.setSign(Plus);
    }

    // Case a,b == Minus Plus
    // f x g == (a-a*)/i x (b+b*)
    //       == a x b + a x b* - a* x b - a* x b*
    //  rtn1 == (a x b   - (a x b)*)/i
    //  rtn2 == (a x b*  - (a* x b))/i
    else if (asign == Minus && bsign == Plus) {
        rtn1.setSign(Minus);
        rtn2.setSign(Minus);
    }

    // Case Plus Minus
    // f x g == (a+a*) x (b-b*)/i
    //       ==  (a x b  - a x b* + a* x b - a* x b*)/i
    //  rtn1 ==  (a x b  -  [a x b]*)/i
    //  rtn2 == (-a x b* - [-a* x b])/i
    else if (asign == Plus && bsign == Minus) {
        rtn1.setSign(Minus);
        rtn2 *= -1;
        rtn2.setSign(Minus);
    }
    // Case Minus Minus
    // f x g == (a-a*)/i x (b-b*)/i
    //       == -(a x b  - a x b* - a* x b + a* x b*)
    //  rtn1 ==  -a x b  +  [-a x b]*
    //  rtn2 ==   a x b* +  [a* x b]
    else {
        rtn1 *= -1;
        rtn1.setSign(Plus);
        rtn2.setSign(Plus);
    }
    rtn1.canonicalize();
    rtn2.canonicalize();

    return;
}

vector<RealProfile> realBasisKxKz(int Ny, int kx, int kz, Real Lx, Real Lz, Real a, Real b, const BasisFlags& flags) {
    assert(Ny > 0);
    assert(Lx > 0 && Lz > 0);
    assert(a < b);

    vector<BasisFunc> complexbasis = complexBasisKxKz(Ny, kx, kz, Lx, Lz, a, b, flags);

    int Nreal;
    int Ncmplx = complexbasis.size();
    vector<RealProfile> realbasis;

    // 0,0 mode is pure real already.
    if (kx == 0 && kz == 0) {
        Nreal = Ncmplx;
        realbasis.reserve(Nreal);
        for (vector<BasisFunc>::iterator n = complexbasis.begin(); n != complexbasis.end(); ++n)
            realbasis.push_back(RealProfile(*n, Plus));
    }
    // Make two real modes from each complex mode
    else {
        Nreal = 2 * Ncmplx;
        realbasis.reserve(Nreal);
        for (vector<BasisFunc>::iterator n = complexbasis.begin(); n != complexbasis.end(); ++n) {
            realbasis.push_back(RealProfile(*n, Minus));
            realbasis.push_back(RealProfile(*n, Plus));
        }
    }
    return realbasis;
}

// Begun on Wed Dec 13 09:53:17 EST 2006
vector<RealProfile> realBasis(int Ny, int kxmax, int kzmax, Real Lx, Real Lz, Real a, Real b, const BasisFlags& flags) {
    assert(0 <= kxmax && 0 <= kzmax);
    assert(Ny > 0);
    assert(Lx > 0 && Lz > 0);
    assert(a < b);

    // The following calculation of the total number of real-valued basis
    // functions is based on the calculation of the number of complex basis
    // functions performed in the complexBasis() function in basisfunc.cpp.
    // Don't change one and not the other. And refer to comments there.

    // Consider the following kx,kz grid of Fourier modes. The letters mark
    // different portions of Fourier modes that have to be treated differently.

    //    0 1 2 3 4 5 6 7 kz
    //  0 a b b b b b b .
    //  1 c e e e e e e .
    //  2 c e e e e e e .
    //  3 c e e e e e e .
    //  4 c e e e e e e .
    //  5 . . . . . . . .
    // -4 f g g g g g g .
    // -3 f g g g g g g .
    // -2 f g g g g g g .
    // -1 f g g g g g g .
    // kx

    // For this grid Nx=10, Nz=8, kxmax=4 and kzmax=6. Cosine modes are marked
    // with '.' and are always set to zero (see Trefethen96).

    // Modes marked 'f' are complex conjugates of modes marked 'c', so we do
    // not need basis functions for those Fourier modes to determine the
    // linearly independent coefficients in the expansion.
    BC aBC = flags.aBC;
    BC bBC = flags.bBC;
    int Nbc = ((aBC == Diri) ? 1 : 0) + ((bBC == Diri) ? 1 : 0);
    int Nbf;
    if (flags.zerodivergence == false) {
        // # complex polynomials per Fourier mode is 3 indpt u,v,w modes times
        // # of polynomials max degree Ny that match BCs,
        int ppfm = 3 * (Ny - Nbc);

        // Two real modes for each complex mode, times # complex polynomials per
        // Fourier mode, times (# a,b,c,e Fourier modes + # g Fourier modes)
        Nbf = 2 * ppfm * ((kzmax + 1) * (kxmax + 1) + kzmax * kxmax);

        // But the 'a' Fourier mode, kx,kz==0,0, is real valued and doesn't need
        // the doubling in the above. Remove 1 Fourier mode times # complex
        // polynomials per Fourier mode.
        Nbf -= 1 * ppfm;
    } else {
        // For zero-div basis funcs, # complex polynomials per Fourier mode is
        // more complicated.

        // ppfm stands for the number of cmplx-valued polynomials per Fourier mode
        int a_ppfm = 0;  // for the 'a' mode: (0,0) Fourier mode
        int b_ppfm = 0;  // for the 'b' and all other modes

        if (Nbc == 0) {
            a_ppfm = 2 * Ny + 1;
            b_ppfm = 2 * Ny;
        } else if (Nbc == 1) {
            a_ppfm = 2 * Ny - 2;
            b_ppfm = 2 * Ny - 3;
        } else if (Nbc == 2) {
            a_ppfm = 2 * Ny - 4;
            b_ppfm = 2 * Ny - 6;
        }

        // Two real modes for each complex mode times
        // plus one real mode per for the 0,0 Fourier
        //    2*b_ppfm*( (blocks a,b,c,e)  +  (block g)  - (block a))
        //+     a_ppfm*(block a)

        Nbf = 2 * b_ppfm * ((kzmax + 1) * (kxmax + 1) + kzmax * kxmax - 1) + a_ppfm * 1;
    }

    vector<RealProfile> f;
    f.reserve(Nbf);
    //  int j=0;

    // blocks a,b,c,e
    for (int kx = 0; kx <= kxmax; ++kx) {
        for (int kz = 0; kz <= kzmax; ++kz) {
            vector<RealProfile> fkxkz = realBasisKxKz(Ny, kx, kz, Lx, Lz, a, b, flags);
            f.insert(f.end(), fkxkz.begin(), fkxkz.end());
        }
    }
    // block g
    for (int kx = -1; kx >= -kxmax; --kx) {
        for (int kz = 1; kz <= kzmax; ++kz) {
            vector<RealProfile> fkxkz = realBasisKxKz(Ny, kx, kz, Lx, Lz, a, b, flags);
            f.insert(f.end(), fkxkz.begin(), fkxkz.end());
        }
    }
    return f;
}

void orthonormalize(vector<RealProfile>& f) {
    // Modified Gram-Schmidt orthogonalization
    // int N=f.size();
    RealProfile fm_tmp;

    for (vector<RealProfile>::iterator m = f.begin(); m != f.end(); ++m) {
        RealProfile& fm = *m;
        fm *= 1.0 / L2Norm(fm);

        int fmkx = fm.kx();
        int fmkz = fm.kz();
        int fmsign = fm.sign();

        // orthogonalize
        for (vector<RealProfile>::iterator n = m + 1; n != f.end(); ++n) {
            RealProfile& fn = *n;

            if (fmkx == fn.kx() && fmkz == fn.kz() && fmsign == fn.sign()) {
                fm_tmp = fm;
                fm_tmp *= L2InnerProduct(fm, fn);
                fn -= fm_tmp;
            }
        }
    }
    return;
}

void checkBasis(const vector<RealProfile>& e, const BasisFlags& flags, bool orthogcheck) {
    int M = e.size();
    cout << M << " elements in basis set" << endl;

    if (flags.orthonormalize) {
        cout << "\nchecking normality..." << endl;
        int bad12 = 0;
        int bad8 = 0;
        int bad4 = 0;
        for (int m = 0; m < M; ++m) {
            Real norm = L2Norm(e[m]);
            Real err = abs(norm - 1.0);
            if (err > 1e-12) {
                e[m].save("badnorm" + i2s(m));
                ++bad12;
            }
            if (err > 1e-8)
                ++bad8;
            if (err > 1e-4)
                ++bad4;
        }
        cout << bad12 << " with abs(L2Norm(em)-1) > 1e-12" << endl;
        cout << bad8 << " with abs(L2Norm(em)-1) > 1e-8" << endl;
        cout << bad4 << " with abs(L2Norm(em)-1) > 1e-4" << endl;
    }

    if (flags.zerodivergence) {
        cout << "\nchecking divergence..." << endl;
        int bad12 = 0;
        int bad8 = 0;
        int bad4 = 0;
        for (int m = 0; m < M; ++m) {
            Real err = divNorm(e[m]);
            if (err > 1e-12) {
                e[m].save("baddiv" + i2s(m));
                ++bad12;
            }
            if (err > 1e-8)
                ++bad8;
            if (err > 1e-4)
                ++bad4;
        }
        cout << bad12 << " with divNorm(em) > 1e-12" << endl;
        cout << bad8 << " with divNorm(em) > 1e-8" << endl;
        cout << bad4 << " with divNorm(em) > 1e-4" << endl;
    }

    if (flags.aBC == Diri || flags.bBC == Diri) {
        cout << "\nchecking boundary conditions..." << endl;
        int bad12 = 0;
        int bad8 = 0;
        int bad4 = 0;

        for (int m = 0; m < M; ++m) {
            Real err = bcNorm(e[m]);
            if (err > 1e-12) {
                e[m].save("badbc" + i2s(m));
                ++bad12;
            }
            if (err > 1e-8)
                ++bad8;
            if (err > 1e-4)
                ++bad4;
        }
        cout << bad12 << " with bcNorm(em) > 1e-12" << endl;
        cout << bad8 << " with bcNorm(em) > 1e-8" << endl;
        cout << bad4 << " with bcNorm(em) > 1e-4" << endl;
    }

    if (flags.orthonormalize && orthogcheck) {
        cout << "\nchecking orthogonality..." << endl;
        int bad12 = 0;
        int bad8 = 0;
        int bad4 = 0;
        for (int m = 1; m < M; ++m) {
            for (int n = 0; n < m; ++n) {
                Real ip = L2IP(e[m], e[n]);
                if (ip > 1e-12)
                    ++bad12;
                if (ip > 1e-8)
                    ++bad8;
                if (ip > 1e-4)
                    ++bad4;
            }
        }
        cout << bad12 << " with L2(e[m],e[n])ip > 1e-12" << endl;
        cout << bad8 << " with L2(e[m],e[n])ip > 1e-8" << endl;
        cout << bad4 << " with L2(e[m],e[n])ip > 1e-4" << endl;
    }
}

ostream& operator<<(ostream& os, Sign s) {
    char c = '\0';
    switch (s) {
        case Minus:
            c = '-';
            break;
        case Plus:
            c = '+';
            break;
    }
    os << c;
    return os;
}

istream& operator>>(istream& is, Sign& s) {
    char c;
    is >> c;
    switch (c) {
        case '-':
            s = Minus;
            break;
        case '+':
            s = Plus;
            break;
        default:
            cerr << "istream& operator>>(istream& is, Sign& s) : bad Sign char: " << c << endl;
            exit(1);
    }
    return is;
}

}  // namespace chflow
