/**
 * some small mathematical conveniences for complex numbers
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_COMPLEXDEFS_H
#define CHANNELFLOW_COMPLEXDEFS_H

#include "cfbasics/mathdefs.h"

// These functions are isolated in their own header file so they can be
// easily excluded, to avoid name clashes when working with other libraries
// (like Octave).

namespace chflow {

inline Complex exp(const Complex& z) { return std::exp(Re(z)) * Complex(cos(Im(z)), sin(Im(z))); }
inline Complex log(const Complex& z) { return Complex(std::log(abs(z)), arg(z)); }

}  // namespace chflow

#endif
