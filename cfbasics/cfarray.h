/**
 * cfarray.h: a templated 1d cfarray class.
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CHANNELFLOW_ARRAY_H
#define CHANNELFLOW_ARRAY_H

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace chflow {

// forward declaration
void write(std::ostream& os, int n);

void read(std::istream& is, int& n);

typedef double Real;
const int REAL_OUTPUT_DIGITS = 17;

template <class T>
class cfarray {
   public:
    using data_container_t = std::vector<T>;

    cfarray(int N = 0);

    cfarray(int N, const T& t);

    bool operator==(const cfarray& a);

    bool operator!=(const cfarray& a);

    void resize(int N);

    void fill(const T& t);

    inline typename data_container_t::reference operator[](int i);

    inline typename data_container_t::const_reference operator[](int i) const;

    cfarray subvector(int offset, int N) const;

    int N() const;

    int length() const;

    const T* pointer() const;  // Efficiency overrules safety in thesis code.
    T* pointer();

    // save and string& ctor form ascii io pair
    // read and write      form binary io pair
    void save(const std::string& filename) const;  // inverse of cfarray(file)
    void binaryDump(std::ostream& os) const;

    void binaryLoad(std::istream& is);

   private:
    std::vector<T> data_;
};

template <class T>
std::ostream& operator<<(std::ostream&, const cfarray<T>& a);

template <class T>
inline typename cfarray<T>::data_container_t::reference cfarray<T>::operator[](int i) {
    assert(i >= 0 && static_cast<unsigned>(i) < data_.size());
    return data_[i];
}

template <class T>
inline typename cfarray<T>::data_container_t::const_reference cfarray<T>::operator[](int i) const {
    assert(i >= 0 && static_cast<unsigned>(i) < data_.size());
    return data_[i];
}

template <class T>
cfarray<T>::cfarray(int N) : data_(N) {}

template <class T>
cfarray<T>::cfarray(int N, const T& t) : data_(N, t) {}

template <class T>
void cfarray<T>::resize(int N) {
    data_.resize(N);
}

template <class T>
void cfarray<T>::fill(const T& t) {
    std::fill(data_.begin(), data_.end(), t);
}

template <class T>
cfarray<T> cfarray<T>::subvector(int offset, int N) const {
    cfarray<T> subvec(N);
    for (int i = 0; i < N; ++i)
        subvec[i] = data_[i + offset];
    return subvec;
}

template <class T>
bool cfarray<T>::operator==(const cfarray& a) {
    return data_ == a.data_;
}

template <class T>
bool cfarray<T>::operator!=(const cfarray& a) {
    return !(*this == a);
}

template <class T>
int cfarray<T>::length() const {
    return data_.size();
}

template <class T>
int cfarray<T>::N() const {
    return data_.size();
}

template <class T>
T* cfarray<T>::pointer() {
    return data_.data();
}

template <class T>
const T* cfarray<T>::pointer() const {
    return data_.data();
}

template <class T>
void cfarray<T>::save(const std::string& filebase) const {
    std::ofstream os(filebase.c_str());
    os << std::scientific << std::setprecision(17);
    os << "% " << data_.size() << " 1\n";
    for (const auto& item : data_) {
        os << item << '\n';
    }
    os.close();
}

template <class T>
std::ostream& operator<<(std::ostream& os, const cfarray<T>& a) {
    int N = a.length();
    char seperator = (N < 10) ? ' ' : '\n';
    os << std::setprecision(17);
    for (int i = 0; i < N; ++i)
        os << a[i] << seperator;
    return os;
}

template <class T>
void cfarray<T>::binaryDump(std::ostream& os) const {
    write(os, data_.size());
    T* d = data_;
    T* end = data_ + data_.size();
    const int size = sizeof(T);
    while (d < end)
        os.write((char*)d++, size);
}

template <class T>
void cfarray<T>::binaryLoad(std::istream& is) {
    auto N = decltype(data_.size())(0);
    read(is, N);

    data_.resize(N);

    // How can this be made to work with endianness and arbitrary types T?
    T* d = data_.data();
    T* end = data_ + data_.size();
    const int size = sizeof(T);
    while (d < end)
        is.read((char*)d++, size);
}

}  // namespace chflow

#endif
