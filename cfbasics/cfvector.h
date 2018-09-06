/**
 * imple vector class for use with BandedTridiag
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_VECTOR_H
#define CHANNELFLOW_VECTOR_H

#include "cfbasics/cfbasics.h"
#include "cfbasics/mathdefs.h"

#include <fftw3.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>

namespace chflow {

template <class T>
struct FFTAllocator {
    using value_type = T;

    T* allocate(std::size_t n) { return static_cast<T*>(fftw_malloc(n * sizeof(T))); }

    void deallocate(T* p, std::size_t) noexcept { fftw_free(p); }
};

template <class T, class U>
bool operator==(const FFTAllocator<T>& a, const FFTAllocator<U>& b) {
    return true;
}

template <class T, class U>
bool operator!=(const FFTAllocator<T>& a, const FFTAllocator<U>& b) {
    return false;
}

class Vector {
   public:
    inline explicit Vector(int N = 0);
    inline explicit Vector(const std::string& filename);
    inline Vector(const Vector& a) = default;
    inline Vector(Vector&& a) = default;
    inline Vector& operator=(const Vector& a) = default;
    inline Vector& operator=(Vector&& a) = default;
    inline ~Vector() = default;

    inline void resize(int N);
    inline void setToZero();

    inline Real& operator[](int i);
    inline Real operator[](int i) const;
    inline Real& operator()(int i);
    inline Real operator()(int i) const;

    inline Vector& operator*=(Real c);
    inline Vector& operator/=(Real c);
    inline Vector& operator+=(Real c);
    inline Vector& operator-=(Real c);
    inline Vector& operator+=(const Vector& c);
    inline Vector& operator-=(const Vector& c);
    inline Vector& dottimes(const Vector& c);
    inline Vector& dotdivide(const Vector& c);

    // Destructive transform methods. Vector must have proper length.
    inline Vector& abs();

    inline Vector subvector(int offset, int N) const;
    inline Vector modularSubvector(int offset, int N) const;

    inline int length() const;           // length/dimension of vector
    inline const Real* pointer() const;  // Efficiency overrules safety in thesis code.
    inline Real* pointer();

    // save and std::string& ctor form ascii io pair   in filebase.asc
    // read and write      form binary io pair
    inline void save(const std::string& filebase) const;

   protected:
    std::vector<Real, FFTAllocator<Real> > data_;
};

// Same as Matlab u(uistart:uiskip:uiend) = v(vistart:viskip:viend)
inline void assign(Vector& u, int uistart, int uistride, int uiend, Vector& v, int vistart, int vistride, int viend);

inline Vector operator*(Real c, const Vector& v);
inline Vector operator+(const Vector& u, const Vector& v);
inline Vector operator-(const Vector& u, const Vector& v);
inline Real operator*(const Vector& u, const Vector& v);
inline bool operator==(const Vector& u, const Vector& v);

inline Vector dottimes(const Vector& u, const Vector& v);
inline Vector dotdivide(const Vector& u, const Vector& v);

// Should change these to l2norm, l1Norm, linfNorm
inline Real L1Norm(const Vector& v);
inline Real L2Norm(const Vector& v);
inline Real L2Norm2(const Vector& v);
inline Real LinfNorm(const Vector& v);

inline Real L1Dist(const Vector& u, const Vector& v);
inline Real L2Dist(const Vector& u, const Vector& v);
inline Real L2Dist2(const Vector& u, const Vector& v);
inline Real LinfDist(const Vector& u, const Vector& v);

inline Real mean(const Vector& v);
inline int maxElemIndex(const Vector& v);
inline Vector vabs(const Vector& v);

inline int maxElemIndex(const Vector& v);  // index of largest magnitude element

inline std::ostream& operator<<(std::ostream&, const Vector& a);

inline Real& Vector::operator[](int i) {
    assert(i >= 0 && static_cast<unsigned>(i) < data_.size());
    return data_[i];
}

inline Real Vector::operator[](int i) const {
    assert(i >= 0 && static_cast<unsigned>(i) < data_.size());
    return data_[i];
}
inline Real& Vector::operator()(int i) {
    assert(i >= 0 && static_cast<unsigned>(i) < data_.size());
    return data_[i];
}

inline Real Vector::operator()(int i) const {
    assert(i >= 0 && static_cast<unsigned>(i) < data_.size());
    return data_[i];
}

inline int Vector::length() const { return data_.size(); }
inline Real* Vector::pointer() { return data_.data(); }
inline const Real* Vector::pointer() const { return data_.data(); }

inline Vector::Vector(int N) : data_(N, 0.0) {}

inline Vector::Vector(const std::string& filebase) {
    std::ifstream is;
    std::string filename = ifstreamOpen(is, filebase, ".asc");
    if (!is) {
        std::cerr << "Vector::Vector(filebase) : can't open file " << filebase << " or " << (filebase + ".asc")
                  << std::endl;
        exit(1);
    }

    // Read in header. Form is "%5 4" for a 5x4 matrix
    char c;
    int M, N;
    is >> c;
    if (c != '%') {
        std::string message("Vector(filebase): bad header in file ");
        message += filename;
        cferror(message);
    }
    is >> M >> N;

    assert(M == 1 || N == 1);
    data_.resize((M > N) ? M : N, 0.0);

    for (auto& item : data_) {
        is >> item;
    }
    is.close();
}

inline void Vector::resize(int newN) { data_.resize(newN); }

inline void Vector::setToZero() { std::fill(data_.begin(), data_.end(), 0.0); }

inline Vector& Vector::operator*=(Real c) {
    for (auto& item : data_) {
        item *= c;
    }
    return *this;
}

inline Vector& Vector::operator/=(Real c) {
    auto cinv = 1.0 / c;
    *this *= cinv;
    return *this;
}

inline Vector& Vector::operator+=(Real c) {
    for (auto& item : data_) {
        item += c;
    }
    return *this;
}

inline Vector& Vector::operator-=(Real c) {
    for (auto& item : data_) {
        item -= c;
    }
    return *this;
}

inline Vector& Vector::operator+=(const Vector& a) {
    assert(a.length() == length());
    for (auto ii = 0u; ii < data_.size(); ++ii) {
        data_[ii] += a.data_[ii];
    }
    return *this;
}

inline Vector& Vector::operator-=(const Vector& a) {
    assert(a.length() == length());
    for (auto ii = 0u; ii < data_.size(); ++ii) {
        data_[ii] -= a.data_[ii];
    }
    return *this;
}

inline Vector& Vector::dottimes(const Vector& a) {
    assert(a.length() == length());
    for (auto ii = 0u; ii < data_.size(); ++ii) {
        data_[ii] *= a.data_[ii];
    }
    return *this;
}

inline Vector& Vector::dotdivide(const Vector& a) {
    assert(a.length() == length());
    for (auto ii = 0u; ii < data_.size(); ++ii) {
        data_[ii] /= a.data_[ii];
    }
    return *this;
}

inline Vector& Vector::abs() {
    for (auto& item : data_) {
        item = fabs(item);
    }
    return *this;
}

inline Vector Vector::subvector(int offset, int N) const {
    Vector subvec(N);
    assert(static_cast<unsigned>(N + offset) <= data_.size());
    for (int i = 0; i < N; ++i)
        subvec[i] = data_[i + offset];
    return subvec;
}

inline Vector Vector::modularSubvector(int offset, int N) const {
    Vector subvec(N);
    for (int i = 0; i < N; ++i)
        subvec[i] = data_[(i + offset) % data_.size()];
    return subvec;
}

inline void Vector::save(const std::string& filebase) const {
    std::string filename(filebase);
    filename += std::string(".asc");
    std::ofstream os(filename.c_str());

    os << std::scientific << std::setprecision(REAL_DIGITS);
    os << "% " << data_.size() << " 1\n";
    for (auto ii = 0u; ii < data_.size(); ++ii) {
        os << std::setw(REAL_IOWIDTH) << data_[ii] << '\n';
    }
    os.close();
}

inline void assign(Vector& u, int uistart, int uistride, int uiend, Vector& v, int vistart, int vistride, int viend) {
    assert(((uistart - uiend) / uistride == (vistart - viend) / vistride));
    int ui, vi;
    for (ui = uistart, vi = vistart; ui < uiend; ui += uistride, vi += vistride)
        u[ui] = v[vi];
}

inline Vector operator*(Real c, const Vector& v) {
    Vector u(v.length());
    int N = u.length();
    for (int i = 0; i < N; ++i)
        u[i] = c * v[i];
    return u;
}

inline Vector operator+(const Vector& u, const Vector& v) {
    Vector w(v.length());
    int N = u.length();
    assert(v.length() == N);
    for (int i = 0; i < N; ++i)
        w[i] = u[i] + v[i];
    return w;
}

inline Vector operator-(const Vector& u, const Vector& v) {
    Vector w(v.length());
    int N = u.length();
    assert(v.length() == N);
    for (int i = 0; i < N; ++i)
        w[i] = u[i] - v[i];
    return w;
}

inline Vector dottimes(const Vector& u, const Vector& v) {
    Vector w(v.length());
    int N = u.length();
    assert(v.length() == N);
    for (int i = 0; i < N; ++i)
        w[i] = u[i] * v[i];
    return w;
}

Vector dotdivide(const Vector& u, const Vector& v) {
    Vector w(v.length());
    int N = u.length();
    assert(v.length() == N);
    for (int i = 0; i < N; ++i)
        w[i] = u[i] / v[i];
    return w;
}

inline Real operator*(const Vector& u, const Vector& v) {
    Real sum = 0.0;
    int N = u.length();
    assert(v.length() == N);
    for (int i = 0; i < N; ++i)
        sum += u[i] * v[i];
    return sum;
}

inline bool operator==(const Vector& u, const Vector& v) {
    int N = u.length();
    if (v.length() != N)
        return false;
    for (int i = 0; i < N; ++i)
        if (u[i] != v[i])
            return false;

    return true;
}

inline Vector vabs(const Vector& v) {
    Vector rtn(v);
    rtn.abs();
    return rtn;
}

inline Real L1Norm(const Vector& v) {
    Real sum = 0.0;
    int N = v.length();
    for (int i = 0; i < N; ++i)
        sum += fabs(v[i]);
    return sum;
}

inline Real L2Norm(const Vector& v) {
    Real sum = 0.0;
    int N = v.length();
    for (int i = 0; i < N; ++i)
        sum += square(v[i]);
    return sqrt(sum);
}

inline Real L2Norm2(const Vector& v) {
    Real sum = 0.0;
    int N = v.length();
    for (int i = 0; i < N; ++i)
        sum += square(v[i]);
    return sum;
}

inline Real LinfNorm(const Vector& v) {
    Real max = 0.0;
    int N = v.length();
    for (int i = 0; i < N; ++i)
        max = Greater(fabs(v[i]), max);
    return max;
}

inline Real L1Dist(const Vector& u, const Vector& v) {
    assert(u.length() == v.length());
    Real sum = 0.0;
    int N = v.length();
    for (int i = 0; i < N; ++i)
        sum += fabs(u[i] - v[i]);
    return sum;
}

inline Real L2Dist(const Vector& u, const Vector& v) {
    assert(u.length() == v.length());
    Real sum = 0.0;
    int N = v.length();
    for (int i = 0; i < N; ++i)
        sum += square(v[i] - u[i]);
    return sqrt(sum);
}

inline Real L2Dist2(const Vector& u, const Vector& v) {
    assert(u.length() == v.length());
    Real sum = 0.0;
    int N = v.length();
    for (int i = 0; i < N; ++i)
        sum += square(u[i] - v[i]);
    return sum;
}

inline Real LinfDist(const Vector& u, const Vector& v) {
    assert(u.length() == v.length());
    Real max = 0.0;
    int N = v.length();
    for (int i = 0; i < N; ++i)
        max = Greater(fabs(v[i] - u[i]), max);
    return max;
}

inline Real mean(const Vector& v) {
    Real sum = 0.0;
    int N = v.length();
    for (int i = 0; i < N; ++i)
        sum += v[i];
    return sum / N;
}

inline int maxElemIndex(const Vector& v) {
    Real max = 0.0;
    int index = 0;
    int N = v.length();
    for (int i = 0; i < N; ++i)
        if (fabs(v[i]) > max) {
            max = fabs(v[i]);
            index = i;
        }
    return index;
}

inline std::ostream& operator<<(std::ostream& os, const Vector& a) {
    int N = a.length();
    char seperator = (N < 10) ? ' ' : '\n';
    for (int i = 0; i < N; ++i) {
        os << a[i];
        os << seperator;
    }
    return os;
}

}  // namespace chflow
#endif
