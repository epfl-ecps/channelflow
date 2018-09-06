/**
 * Adaption of gls brent.c to nsolver
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2007 Brian Gough, 2015 Tobias Kreilos
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 */

#ifndef NSOLVER_BRENT_H
#define NSOLVER_BRENT_H

#include <functional>

#include <cfbasics/cfbasics.h>

namespace chflow {

class Brent {
   public:
    Brent(std::function<Real(Real)>& f, const Real x_minimum, Real f_minimum, Real x_lower, Real f_lower, Real x_upper,
          Real f_upper);
    void iterate();
    Real minimize(int maxIter, Real epsabs, Real epsrel);
    Real fmin() { return f_minimum_; }
    Real xmin() { return x_minimum_; }
    bool converged(Real epsabs, Real epsrel);

   private:
    std::function<Real(Real)> f_;
    const Real golden = (3. - sqrt(5.)) / 2.;

    Real d, e, v, w;
    Real f_v, f_w;

    Real x_minimum_, x_lower_, x_upper_;
    Real f_minimum_, f_lower_, f_upper_;
};

inline Brent::Brent(std::function<Real(Real)>& f, Real x_minimum, Real f_minimum, Real x_lower, Real f_lower,
                    Real x_upper, Real f_upper)
    : f_(f),
      x_minimum_(x_minimum),
      x_lower_(x_lower),
      x_upper_(x_upper),
      f_minimum_(f_minimum),
      f_lower_(f_lower),
      f_upper_(f_upper) {
    // error checking
    if (x_minimum <= x_lower)
        cferror("Brent constructor: x_minimum <= x_lower");
    if (x_minimum >= x_upper)
        cferror("Brent constructor: x_minimum >= x_upper");
    if (f_minimum >= f_lower)
        cferror("Brent constructor: f_minimum >= f_lower");
    if (f_minimum >= f_upper)
        cferror("Brent constructor: f_minimum >= f_upper");

    v = x_lower + golden * (x_upper - x_lower);
    w = v;
    d = 0;
    e = 0;

    f_v = f_(v);
    f_w = f_v;
}

inline void Brent::iterate() {
    Real u, f_u;

    const Real w_lower = (x_minimum_ - x_lower_);
    const Real w_upper = (x_upper_ - x_minimum_);

    const Real tolerance = 2e-16 * fabs(x_minimum_);

    Real p = 0, q = 0, r = 0;

    const Real midpoint = 0.5 * (x_lower_ + x_upper_);

    if (fabs(e) > tolerance) {
        /* fit parabola */

        r = (x_minimum_ - w) * (f_minimum_ - f_v);
        q = (x_minimum_ - v) * (f_minimum_ - f_w);
        p = (x_minimum_ - v) * q - (x_minimum_ - w) * r;
        q = 2 * (q - r);

        if (q > 0) {
            p = -p;
        } else {
            q = -q;
        }

        r = e;
        e = d;
    }

    if (fabs(p) < fabs(0.5 * q * r) && p < q * w_lower && p < q * w_upper) {
        Real t2 = 2 * tolerance;

        d = p / q;
        u = x_minimum_ + d;

        if ((u - x_lower_) < t2 || (x_upper_ - u) < t2) {
            d = (x_minimum_ < midpoint) ? tolerance : -tolerance;
        }
    } else {
        e = (x_minimum_ < midpoint) ? x_upper_ - x_minimum_ : -(x_minimum_ - x_lower_);
        d = golden * e;
    }

    if (fabs(d) >= tolerance) {
        u = x_minimum_ + d;
    } else {
        u = x_minimum_ + ((d > 0) ? tolerance : -tolerance);
    }

    // Evaluate function
    f_u = f_(u);

    if (f_u <= f_minimum_) {
        if (u < x_minimum_) {
            x_upper_ = x_minimum_;
            f_upper_ = f_minimum_;
        } else {
            x_lower_ = x_minimum_;
            f_lower_ = f_minimum_;
        }

        v = w;
        f_v = f_w;
        w = x_minimum_;
        f_w = f_minimum_;
        x_minimum_ = u;
        f_minimum_ = f_u;
        return;
    } else {
        if (u < x_minimum_) {
            x_lower_ = u;
            f_lower_ = f_u;
        } else {
            x_upper_ = u;
            f_upper_ = f_u;
        }

        if (f_u <= f_w || w == x_minimum_) {
            v = w;
            f_v = f_w;
            w = u;
            f_w = f_u;
            return;
        } else if (f_u <= f_v || v == x_minimum_ || v == w) {
            v = u;
            f_v = f_u;
            return;
        }
    }

    return;
}

inline Real Brent::minimize(int maxIter, Real epsabs, Real epsrel) {
    int nIter = 0;
    while (nIter < maxIter && converged(epsabs, epsrel) == false) {
        nIter++;
        iterate();
    }
    return x_minimum_;
}

// copy of gsl_min_test_interval
inline bool Brent::converged(Real epsabs, Real epsrel) {
    const double abs_lower = fabs(x_lower_);
    const double abs_upper = fabs(x_upper_);

    double min_abs, tolerance;

    if (epsrel < 0.0)
        cferror("relative tolerance is negative");

    if (epsabs < 0.0)
        cferror("absolute tolerance is negative");

    if (x_lower_ > x_upper_)
        cferror("lower bound larger than upper_bound");

    if ((x_lower_ > 0 && x_upper_ > 0) || (x_lower_ < 0 && x_upper_ < 0)) {
        min_abs = abs_lower < abs_upper ? abs_lower : abs_upper;
    } else {
        min_abs = 0;
    }

    tolerance = epsabs + epsrel * min_abs;

    if (fabs(x_upper_ - x_lower_) < tolerance)
        return true;

    return false;
}

}  // namespace chflow

#endif
