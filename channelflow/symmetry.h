/**
 * The symmetry group for 3D FlowFields
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_SYMMETRY_H
#define CHANNELFLOW_SYMMETRY_H

#include <fftw3.h>
#include "cfbasics/cfvector.h"
#include "cfbasics/mathdefs.h"
#include "channelflow/basisfunc.h"
#include "channelflow/chebyshev.h"
#include "channelflow/realprofile.h"

namespace chflow {

// FieldSymmetry sigma(sx,sy,sz,ax,az,s) takes ax,az in [0,1] and s,sx,sy,sz in {-1,1}

// scalar f, vector (u,v,w), tensor f(u,v,w)
// sigma : f(x,y,z) -> s f(sx x + ax Lx, sy y, sz z + az Lz)
// sigma : (u,v,w)(x,y,z) -> s (sx u, sy v, sz w)(sx x + ax Lx, sy y, sz z + az Lz)

class FlowField;

class FieldSymmetry {
   public:
    FieldSymmetry();
    // FieldSymmetry(std::istream& is);
    FieldSymmetry(const std::string& filebase);
    FieldSymmetry(int s);             // sign change in (u,v,w), s = +/-1
    FieldSymmetry(Real ax, Real az);  // pure translate
    FieldSymmetry(int sx, int sy, int sz, Real ax = 0.0, Real az = 0.0, int s = 1);
    FieldSymmetry(bool sx, bool sy, bool sz, Real ax = 0.0, Real az = 0.0, bool s = false);

    FlowField operator()(const FlowField& u) const;
    RealProfile operator()(const RealProfile& u) const;
    FieldSymmetry& operator*=(const FieldSymmetry& s);  // (*this) = s * (*this), non-commutative!
    FieldSymmetry& operator*=(Real c);                  // ax,az *= c;
    // note: projection operators are external functions, declared below

    inline int s() const;
    inline int sx() const;
    inline int sy() const;
    inline int sz() const;
    inline Real ax() const;
    inline Real az() const;
    int s(int i) const;
    int sign(int i) const;

    void save(const std::string& filebase, std::ios::openmode openflag = std::ios::out) const;

    bool isIdentity() const;  // true if FieldSymmetry is "1 1 1 1 0 0"

   private:
    int s_;    // (u,v,w)(x,y,z) -> s (u, v, w)(x,y,z)
    int sx_;   // (u,v,w)(x,y,z) -> (sx u, v,w)(sx x, y,z)
    int sy_;   // (u,v,w)(x,y,z) -> (u, sy v,w)(x, sy y,z)
    int sz_;   // (u,v,w)(x,y,z) -> (u,v, sz w)(x,y, sz z)
    Real ax_;  // (u,v,w)(x,y,z) -> (u,v,w)(x+ax*Lx,y,z)
    Real az_;  // (u,v,w)(x,y,z) -> (u,v,w)(x,y,z+az/Lz)
};

bool operator==(const FieldSymmetry& p, const FieldSymmetry& q);
bool operator!=(const FieldSymmetry& p, const FieldSymmetry& q);

std::istream& operator>>(std::istream& is, FieldSymmetry& s);
std::ostream& operator<<(std::ostream& os, const FieldSymmetry& s);

// FieldSymmetry optimizePhase(const FlowField& u, const FieldSymmetry& s);
FieldSymmetry operator*(const FieldSymmetry& p, const FieldSymmetry& q);  // (p*q)(u) = p(q(u))
FieldSymmetry inverse(const FieldSymmetry& s);
FlowField operator*(const FieldSymmetry& s, const FlowField& u);  // same as s(u)

class SymmetryList : public cfarray<FieldSymmetry> {
   public:
    SymmetryList();
    SymmetryList(int n);
    SymmetryList(const std::string& filebase);

    void save(const std::string& filebase) const;
};

std::ostream& operator<<(std::ostream& os, const SymmetryList& s);

void project(const FieldSymmetry& s, const FlowField& u, FlowField& Pu);  // Pu = 1/2 (1+s) u
FlowField project(const FieldSymmetry& s, FlowField& u);                  // return  1/2 (1+s) u

void project(const cfarray<FieldSymmetry>& s, const FlowField& u, FlowField& Pu);
FlowField project(const cfarray<FieldSymmetry>& s, FlowField& u);  // return 1/2^n (1+s[0])(1+s[1]) ... u

// Project u onto invariant subspace and warn if L2Dist(u, Pu) > eps
void project(const cfarray<FieldSymmetry>& s, FlowField& u, const std::string& label, std::ostream& os,
             Real eps = 1e-8);

inline Real FieldSymmetry::ax() const { return ax_; }
inline Real FieldSymmetry::az() const { return az_; }
inline int FieldSymmetry::s() const { return s_; }
inline int FieldSymmetry::sx() const { return sx_; }
inline int FieldSymmetry::sy() const { return sy_; }
inline int FieldSymmetry::sz() const { return sz_; }

FieldSymmetry operator*(const FieldSymmetry& s1, const FieldSymmetry& s2);

FieldSymmetry quadraticInterpolate(const cfarray<FieldSymmetry>& sn, const cfarray<Real>& xn, Real x);
FieldSymmetry polynomialInterpolate(const cfarray<FieldSymmetry>& sn, const cfarray<Real>& xn, Real x);
}  // namespace chflow

#endif
