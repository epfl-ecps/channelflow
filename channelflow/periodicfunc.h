/**
 * Fourier representation of periodic functions
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_PERIODICFUNC_H
#define CHANNELFLOW_PERIODICFUNC_H

#include "cfbasics/cfvector.h"
#include "cfbasics/mathdefs.h"

#include <fftw3.h>

#include <iostream>
#include <memory>

namespace chflow {

Vector periodicpoints(int N, Real L);  // N+1 pts {0, L/N, 2L/N, ..., L}

// A class for representing 1d periodic functions on the interval [0, L]
class PeriodicFunc {
   public:
    PeriodicFunc();
    PeriodicFunc(const PeriodicFunc& f);
    PeriodicFunc(uint N, Real L, fieldstate s = Spectral, uint flag = FFTW_ESTIMATE);
    PeriodicFunc(const std::string& filebase);  // read ascii from file
    PeriodicFunc& operator=(const PeriodicFunc& f);

    inline Real operator()(uint n) const;  // gridpoint value f(x_n)
    inline Real& operator()(uint n);       // gridpoint value f(x_n)
    inline Complex cmplx(uint k) const;    // fourier coefficient f_k
    inline Complex& cmplx(uint k);         // fourier coefficient f_k
    inline Real operator[](uint n) const;  // nth data cfarray value
    inline Real& operator[](uint n);       // nth data cfarray value
    inline Real x(uint n) const;           // position of nth gridpt x_n

    void save(const std::string& filebase, fieldstate s = Physical) const;

    void reconfig(const PeriodicFunc& f);
    void randomize(Real magn, Real decay);
    void setLength(Real L);
    void setState(fieldstate s);
    void setToZero();
    void resize(uint N, Real L);

    Real operator()(Real x) const;
    Real eval(Real x) const;

    // void eval(const Vector& x, PeriodicFunc& g) const;
    // PeriodicFunc eval(const Vector& x) const;
    // void binaryDump(std::ostream& os) const;
    // void binaryLoad(std::istream& is);
    // void fill(const PeriodicFunc& g);
    // void interpolate(const PeriodicFunc& g); // *this is on subdomain of g.
    // void reflect(const PeriodicFunc& g, parity p); // reflect g about u.a()
    // inline uint Ngridpts() const; // same as N

    inline Real L() const;
    inline uint N() const;
    inline uint Npad() const;    // 2*(N/2+1)
    inline uint Nmodes() const;  // N/2+1
    inline uint kmax() const;    // N/2 (note: N/2 mode for N even is set to 0)

    inline fieldstate state() const;
    Real mean() const;

    PeriodicFunc& operator*=(Real c);
    PeriodicFunc& operator+=(const PeriodicFunc& g);
    PeriodicFunc& operator-=(const PeriodicFunc& g);
    PeriodicFunc& operator*=(const PeriodicFunc& g);  // dottimes, only for Physical

    void fft();                    // transform from Physical to Spectral
    void ifft();                   // transform from Spectral to Physcial
    void makeSpectral();           // if Physical, transform to Spectral
    void makePhysical();           // if Spectral, transform to Physical
    void makeState(fieldstate s);  // if state != s, transform to state s

    bool congruent(const PeriodicFunc& g) const;

    friend void swap(PeriodicFunc& f, PeriodicFunc& g);

   private:
    uint N_ = 0;                                          // number of gridpoints
    Real L_ = 0;                                          // upper bound of domain
    std::unique_ptr<void, void (*)(void*)> data_handle_;  // Manages the lifetime of FFTW data buffer
    Real* rdata_ = nullptr;                               // cfarray for Fourier coeffs or physical data
    Complex* cdata_ = nullptr;                            // Complex alias for rdata_
    fieldstate state_;                                    // indicates Physical or Spectral state of object

    void fftw_initialize(uint fftw_flags = FFTW_ESTIMATE);
    uint fftw_flags_;
    std::unique_ptr<std::remove_pointer<fftw_plan>::type, void (*)(fftw_plan)> forward_plan_;
    std::unique_ptr<std::remove_pointer<fftw_plan>::type, void (*)(fftw_plan)> inverse_plan_;
};

PeriodicFunc operator*(Real c, const PeriodicFunc& g);
PeriodicFunc operator+(const PeriodicFunc& f, const PeriodicFunc& g);
PeriodicFunc operator-(const PeriodicFunc& f, const PeriodicFunc& g);
bool operator==(const PeriodicFunc& f, const PeriodicFunc& g);
bool operator!=(const PeriodicFunc& f, const PeriodicFunc& g);

void diff(const PeriodicFunc& f, PeriodicFunc& df);
void diff2(const PeriodicFunc& f, PeriodicFunc& d2f);
void diff2(const PeriodicFunc& f, PeriodicFunc& d2f, PeriodicFunc& tmp);
void diff(const PeriodicFunc& f, PeriodicFunc& df, uint n);
PeriodicFunc diff(const PeriodicFunc& f);
PeriodicFunc diff2(const PeriodicFunc& f);
PeriodicFunc diff(const PeriodicFunc& f, uint n);

// Integrate sets the arbitrary const of integration so that mean(u)==0.
void integrate(const PeriodicFunc& df, PeriodicFunc& f);
PeriodicFunc integrate(const PeriodicFunc& df);

// Find a zero of f(x) via Newton search.
Real newtonSearch(const PeriodicFunc& f, Real xguess, int Nmax = 10, Real eps = 1e-12);

//   normalize  ?   false                  :  true
// L2Norm2(f)   =      Int_a^b f^2 dy            1/(b-a) Int_a^b f^2 dy
// L2Dist2(f,g) =      Int_a^b (f-g)^2 dy        1/(b-a) Int_a^b (f-g)^2 dy
// L2Norm(f)    = sqrt(Int_a^b f^2 dy)      sqrt(1/(b-a) Int_a^b f^2 dy)
// L2Dist(f,g)  = sqrt(Int_a^b (f-g)^2 dy)  sqrt(1/(b-a) Int_a^b (f-g)^2 dy
// L2IP...(f,g) =      Int_a^b f g dy            1/(b-a) Int_a^b f g dy
//
// L2IP is L2 Inner Product

Real L2Norm(const PeriodicFunc& f, bool normalize = true);
Real L2Norm2(const PeriodicFunc& f, bool normalize = true);
Real L2Dist(const PeriodicFunc& f, const PeriodicFunc& g, bool normalize = true);
Real L2Dist2(const PeriodicFunc& f, const PeriodicFunc& g, bool normalize = true);
Real L2IP(const PeriodicFunc& f, const PeriodicFunc& g, bool normalize = true);

// Vector zeros(const PeriodicFunc& f, Real epsilon=1e-14);

std::ostream& operator<<(std::ostream& os, const PeriodicFunc& f);

inline Real PeriodicFunc::L() const { return L_; }
inline uint PeriodicFunc::N() const { return N_; }
inline uint PeriodicFunc::Npad() const { return 2 * (N_ / 2 + 1); }
inline uint PeriodicFunc::Nmodes() const { return (N_ / 2 + 1); }
inline uint PeriodicFunc::kmax() const { return N_ / 2; }
inline fieldstate PeriodicFunc::state() const { return state_; }

inline Real PeriodicFunc::operator()(uint n) const {
    assert(state_ == Physical);
    assert(n < N_);
    return rdata_[n];
}

inline Real& PeriodicFunc::operator()(uint n) {
    assert(state_ == Physical);
    assert(n < N_);
    return rdata_[n];
}

inline Complex PeriodicFunc::cmplx(uint k) const {
    assert(state_ == Spectral);
    assert(k < Nmodes());
    return cdata_[k];
}

inline Complex& PeriodicFunc::cmplx(uint k) {
    assert(state_ == Spectral);
    assert(k < Nmodes());
    return cdata_[k];
}

inline Real PeriodicFunc::operator[](uint n) const {
    assert(n < Npad());
    return rdata_[n];
}

inline Real& PeriodicFunc::operator[](uint n) {
    assert(n < Npad());
    return rdata_[n];
}

inline Real PeriodicFunc::x(uint n) const { return n * (L_ / N_); }

}  // namespace chflow

#endif
