/**
 * Basic functions and definitions for the Channelflow library
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#ifndef CFBASICS_H
#define CFBASICS_H

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include <arpa/inet.h>
#include <fftw3.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef HAVE_MPI
#include "mpi.h"
#endif

#include "Eigen/Dense"
#include "cfbasics/cfarray.h"
#include "cfbasics/mathdefs.h"

extern const char* g_GIT_SHA1;

namespace chflow {

enum HookstepPhase { ConstantDelta, ReducingDelta, IncreasingDelta, Finished };
enum ResidualImprovement { Unacceptable, Poor, Ok, Good, Accurate, NegaCurve };
// Newton algorithm enums:
enum SolverMethod { SolverEigen, SolverGMRES, SolverFGMRES, SolverBiCGStab };
enum OptimizationMethod { None, Linear, Hookstep };
enum SolutionType { Equilibrium, PeriodicOrbit };
// This enum is used for labeling diagnostic output data
enum fEvalType { fEval, DfEval, HookstepEval };

using Rn2Rnfunc = std::function<Eigen::VectorXd(Eigen::VectorXd)>;

inline std::string r2s(Real r);
inline std::string i2s(int n, int length = 0, char pad = '0');
inline void cferror(const std::string& message);
inline void cfpause();
inline bool fileExists(const std::string& filename);
inline std::string FillZeros(int i, int n);  // FillZeros(12, 5) = "00012"
inline std::string FillZeros(Real t, int n);
inline std::string pwd();
// inline void error (const std::string& message);

inline int mpirank();  // return rank if we are using MPI, 0 otherwise

// From foo.ext or foo return foo
inline std::string removeSuffix(const std::string& filename, const std::string& extension);
// From foo.ext or foo return foo.ext
inline std::string appendSuffix(const std::string& filename, const std::string& extension);

// Does filename have the given extension?
inline bool hasSuffix(const std::string& filename, const std::string& extension);

inline bool isReadable(const std::string& filename);

// Attempt to open filebase.ext then filebase, return successful filename
// eg string filename = ifstreamOpen(is, "foo", ".bin", ios::binary);
inline std::string ifstreamOpen(std::ifstream& is, const std::string& filebase, const std::string& ext,
                                std::ios_base::openmode mode = std::ios::in);
// Ascii IO
inline void save(Real c, const std::string& filebase);
inline void load(Real& c, const std::string& filebase);
inline void save(Complex c, const std::string& filebase);
inline void load(Complex& c, const std::string& filebase);

// Binary IO
inline void write(std::ostream& os, int n);
inline void write(std::ostream& os, bool b);
inline void write(std::ostream& os, Real x);
inline void write(std::ostream& os, Complex z);
inline void write(std::ostream& os, fieldstate s);

inline void read(std::istream& is, int& n);
inline void read(std::istream& is, bool& b);
inline void read(std::istream& is, Real& x);
inline void read(std::istream& is, Complex& z);
inline void read(std::istream& os, fieldstate& s);

// end mathdefs port

inline void print(const Eigen::MatrixXd& x);
inline void print(const Eigen::VectorXd& x);
inline void save(const Eigen::MatrixXd& A, const std::string& filebase);
inline void save(const Eigen::VectorXd& x, const std::string& filebase);
inline void save(const Eigen::VectorXcd& x, const std::string& filebase);
inline void load(Eigen::MatrixXd& A, const std::string& filebase);
inline void load(Eigen::VectorXd& x, const std::string& filebase);
inline void load(Eigen::VectorXcd& x, const std::string& filebase);
inline Real getRealfromLine(int taskid, std::ifstream& is);
inline int getIntfromLine(int taskid, std::ifstream& is);
inline std::string getStringfromLine(int taskid, std::ifstream& is);
inline void setToZero(Eigen::MatrixXd& A);
inline void setToZero(Eigen::VectorXd& x);

inline Real L2Norm(const Eigen::VectorXd& x, int cutoff = 0);
inline Real L2Norm(const Eigen::VectorXcd& x, int cutoff = 0);
inline Real L2Norm2(const Eigen::VectorXd& x, int cutoff = 0);
inline Real L2Norm2(const Eigen::VectorXcd& x, int cutoff = 0);
inline Real L2Dist(const Eigen::VectorXd& x, const Eigen::VectorXd& y, int cutoff = 0);
inline Real L2Dist(const Eigen::VectorXcd& x, const Eigen::VectorXcd& y, int cutoff = 0);
inline Real L2Dist2(const Eigen::VectorXd& x, const Eigen::VectorXd& y, int cutoff = 0);
inline Real L2Dist2(const Eigen::VectorXcd& x, const Eigen::VectorXcd& y, int cutoff = 0);

inline Real L2IP(const Eigen::VectorXd& u, const Eigen::VectorXd& v, int cutoff = 0);
inline void operator*=(Eigen::VectorXcd& x, Complex c);

inline void rename(const std::string& oldname, const std::string& newname);
inline Real pythag(Real a, Real b);
inline Real linearInterpolate(Real x0, Real y0, Real x1, Real y1, Real x);

inline std::ostream& operator<<(std::ostream& os, HookstepPhase p);
inline std::ostream& operator<<(std::ostream& os, ResidualImprovement i);
inline std::ostream& operator<<(std::ostream& os, SolutionType solntype);

inline void rescale(Eigen::VectorXd& x, const Eigen::VectorXd& xscale);  // x -> x./xscale
inline void unscale(Eigen::VectorXd& x, const Eigen::VectorXd& xscale);  // x -> x.*xscale

inline Real adjustDelta(Real delta, Real rescale, Real deltaMin, Real deltaMax, std::ostream& os = std::cout);
inline Real adjustLambda(Real lambda, Real lambdaMin, Real lambdaMax, std::ostream& os = std::cout);

inline void solve(const Eigen::MatrixXd& Ut, const Eigen::VectorXd& D, const Eigen::MatrixXd& C,
                  const Eigen::VectorXd& b, Eigen::VectorXd& x);

inline Real align(const Eigen::VectorXd& u, const Eigen::VectorXd& v);

// Measure execution time (wall-clock time) in seconds.
// Returns seconds passed since first call
inline Real executionTime();

inline void mkdir(const std::string& dirname);

// Neville algorithm quadratic inter/extrapolate based on f[0],f[1],f[2] and x[0],x[1],x[2]
inline Real quadraticInterpolate(const cfarray<Real>& fn, const cfarray<Real>& xn, Real x);
inline Real linearInterpolate(Real x0, Real f0, Real x1, Real f1, Real x);
inline bool isconst(cfarray<Real> f, Real eps = 1e-13);

// Neville algorithm (N-1)th order inter/extrapolate based on f[0],f[1],...,f[N-1] and x[0],x[1],...
inline Real polynomialInterpolate(const cfarray<Real>& fn, const cfarray<Real>& xn, Real x);

// secantSearch and bisectSearch:
// Find root of f(x) between a and b to tolerance feps from polynomial
// interpolant of {fn = f(xn)} I.e. rtn x st std::abs(f(x)) < feps and a <= x <= b.
inline Real secantSearch(Real a, Real b, cfarray<Real>& fn, const cfarray<Real>& xn, Real feps = 1e-14,
                         int maxsteps = 50);

inline Real bisectSearch(Real a, Real b, cfarray<Real>& fn, const cfarray<Real>& xn, Real feps = 1e-14,
                         int maxsteps = 50);

// Make directory and set permissions to a+rx
inline void mkdir(const std::string& dirname);
inline void rename(const std::string& oldname, const std::string& newname);

// Append trailing / if necess
inline std::string pathfix(const std::string& path);

// Convert time to string in format convenient for filenames.
// Returns 1 or 1.000 or 1.250 as appropriate. Three digits
inline std::string t2s(Real t, bool decimals);

// Return false, approx, or true.
inline std::string fuzzyless(Real x, Real eps);

inline std::string pwd();

// clip: from string "foo.asc" or "foo" return "foo"
// stub: from string "/home/foo.asc" or "foo.asc" return "foo.asc"
inline std::string clip(const std::string& filename, const std::string& ext);
inline std::string stub(const std::string& filename, const std::string& ext);

inline std::string FillZeros(int i, int n);  // FillZeros(12, 5) = "00012"
inline std::string FillZeros(Real t, int n);
// inline std::string FillSpaces (int i, int n); // FillSpaces(12, 5) = "   12"
// inline std::string FillSpaces (Real t, int n);

template <class T>
inline void push(const T& t, cfarray<T>& cfarrayt);

inline void openfile(std::ofstream& f, std::string filename, std::ios::openmode openflag = std::ios::out);

inline bool checkFlagContent(std::ifstream& is, const std::vector<std::string>& flagList);
// ++++++++++++++++++++++++++++++++++++++++++++
// begin definitions
// ++++++++++++++++++++++++++++++++++++++++++++

inline void printout(const std::string& message, int taskid, bool newline = true, std::ostream& os = std::cout) {
    if (taskid == 0) {
        os << message;
        if (newline)
            os << std::endl;
        os << std::flush;
    }
}

inline void printout(const std::string& message, bool newline = true, std::ostream& os = std::cout) {
    int taskid = 0;
#ifdef HAVE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized)
        MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
#endif
    printout(message, taskid, newline, os);
}

inline void printout(const std::string& message, std::ostream& os) { printout(message, true, os); }

inline void printout(const std::stringstream& sstr, std::ostream& os = std::cout) { printout(sstr.str(), false, os); }

inline void printToFile(const std::string& message, const std::string& filename, bool newline = true) {
    int taskid = 0;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
#endif
    if (taskid == 0) {
        std::ofstream f(filename.c_str(), std::ios::app);
        f << message;
        if (newline)
            f << std::endl;
        f.close();
    }
}

/** Returns the extension of a file passed as argument
 *
 * \param[in] filename the input file
 *
 * \return the file extension (without dot)
 */
std::string getfileextension(std::string filename);

/** Returns the base of a filename
 *
 * \param[in] filename the input files
 *
 * \return filename stripped of path and extension
 */
std::string getfilebase(std::string filename);

// things ported from channelflow/mathdefs.cpp

inline int mpirank() {
    int rank = 0;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    return rank;
}

inline bool haveMPI() {
#ifdef HAVE_MPI
    return true;
#else
    return false;
#endif
}

inline void cferror(const std::string& message) {
    std::cout << message << std::endl;
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    exit(1);
}

inline void cfpause() {
    if (mpirank() == 0) {
        std::cout << "cfpause..." << std::flush;
        char s[10];
        std::cin.getline(s, 10);
    }
}

inline std::string r2s(Real r) {
    const int Nbuf = 32;
    char buff[Nbuf];
    sprintf(buff, "%g", r);
    return std::string(buff);
}

inline std::string i2s(int n, int length, char pad) {
    //   const int Nbuf = 32;
    //   char buff[Nbuf];
    //   sprintf (buff, "%d", n);
    //   return string (buff);
    std::stringstream ss;
    ss << n;
    std::string s = ss.str();
    int l = s.length();
    for (int j = l; j < length; ++j) {
        s = pad + s;
    }
    return s;
}

// Return filebase.extension unless filebase already includes .extension
inline std::string appendSuffix(const std::string& filebase, const std::string& extension) {
    int Lbase = filebase.length();
    int Lext = extension.length();
    std::string filename = filebase;
    if (Lbase < Lext || filebase.substr(Lbase - Lext, Lext) != extension)
        filename += extension;
    return filename;
}

// If filename == foo.ext, return foo, else return filename
inline std::string removeSuffix(const std::string& filename, const std::string& extension) {
    int extpos = filename.find(extension);
    if (extpos == -1)
        extpos = filename.length();
    return std::string(filename, 0, extpos);
}

inline bool hasSuffix(const std::string& filename, const std::string& extension) {
    int extpos = filename.find(extension);
    if (extpos == -1)
        return false;
    else
        return true;
}

inline bool isReadable(const std::string& filename) {
    std::ifstream is(filename.c_str(), std::ios::in | std::ios::binary);
    bool rtn = (is) ? true : false;
    is.close();
    return rtn;
}

inline std::string ifstreamOpen(std::ifstream& is, const std::string& filebase, const std::string& ext,
                                std::ios_base::openmode mode) {
    mode = mode | std::ios::in;

    // Try to open <filebase>
    std::string filename = filebase;

    is.open(filename.c_str(), mode);

    if (is)
        return filename;

    is.close();
    is.clear();
    filename += ext;
    is.open(filename.c_str(), mode);

    if (!is)
        is.close();

    return filename;
}

// ====================================================================
// Endian-independent binary IO code (i.e. this code is indpt of endianness)
inline void write(std::ostream& os, bool b) {
    char c = (b) ? '1' : '0';
    os.write(&c, 1);  // sizeof(char) == 1
}

inline void read(std::istream& is, bool& b) {
    char c;
    is.read(&c, 1);
    b = (c == '0') ? false : true;
}

inline void write(std::ostream& os, fieldstate s) {
    char c = (s == Spectral) ? 'S' : 'P';
    os.write(&c, 1);
}

inline void read(std::istream& is, fieldstate& s) {
    char c;
    is.read(&c, 1);
    s = (c == 'S') ? Spectral : Physical;
}

inline void write(std::ostream& os, Complex z) {
    Real x = Re(z);
    write(os, x);
    x = Im(z);
    write(os, x);
}

inline void read(std::istream& is, Complex& z) {
    Real a, b;
    read(is, a);
    read(is, b);
    z = Complex(a, b);
}
// ====================================================================
// Binary IO: native saved to big-endian format on disk.
//
// Cross-endian IO assumes 32-bit int.
inline void write(std::ostream& os, int n) {
    if (sizeof(int) != 4) {
        std::cerr << "write(ostream& os, int n) :  channelflow binary IO assumes 32-bit int\n";
        std::cerr << "The int on this platform is " << 8 * sizeof(int) << " bits" << std::endl;
        exit(1);
    }

    n = htonl(n);
    os.write((char*)&n, sizeof(int));
}

inline void read(std::istream& is, int& n) {
    if (sizeof(int) != 4) {
        std::cerr << "read(istream& is, int n) error: channelflow binary IO assumes 32-bit int.\n";
        std::cerr << "The int on this platform is " << 8 * sizeof(int) << " bits" << std::endl;
        exit(1);
    }

    is.read((char*)(&n), sizeof(int));
    n = ntohl(n);
}

// Linux has a bswap_64 function, but the following is more portable.
inline void write(std::ostream& os, Real x) {
    if (sizeof(double) != 8) {
        std::cerr << "write(ostream& os, Real x) error: channelflow binary IO assumes 64-bit double-precision floating "
                     "point.\n";
        std::cerr << "The double on this platform is " << 8 * sizeof(Real) << " bits" << std::endl;
        exit(1);
    }

    if (sizeof(int) != 4) {
        std::cerr << "write(ostream& os, Real x) error: channelflow binary IO assumes 64-bit double-precision floating "
                     "point and 32-bit ints.\n";
        std::cerr << "The int on this platform is " << 8 * sizeof(int) << " bits" << std::endl;
        exit(1);
    }

#if __BYTE_ORDER == __LITTLE_ENDIAN
    int i[2];

    // Copy 64 bit x into two consecutive 32 bit ints
    memcpy(i, &x, sizeof(uint64_t));

    // Byteswap each 32 bit int and reverse their order.
    int tmp = i[1];
    i[1] = htonl(i[0]);
    i[0] = htonl(tmp);

    // Write out the two reversed and byteswapped 32bit ints
    os.write((char*)i, sizeof(double));
#else
    os.write((char*)&x, sizeof(double));
#endif
}

inline void read(std::istream& is, Real& x) {
    if (sizeof(double) != 8) {
        std::cerr
            << "read(istream& is, Real x) error: nsolver binary IO assumes 64-bit double-precision floating point.\n";
        std::cerr << "The double on this platform is " << 8 * sizeof(Real) << " bits" << std::endl;
        exit(1);
    }

    if (sizeof(int) != 4) {
        std::cerr
            << "write(ostream& os, Real x) error: nsolver binary IO assumes 64-bit double-precision floating point "
               "and 32-bit ints.\n";
        std::cerr << "The int on this platform is " << 8 * sizeof(int) << " bits" << std::endl;
        exit(1);
    }

#if __BYTE_ORDER == __LITTLE_ENDIAN
    int i[2];

    // Read in two reversed and byteswapped 32bit ints
    is.read((char*)(i), sizeof(double));

    // Byteswap each 32 bit int and reverse their order
    int tmp = i[1];
    i[1] = ntohl(i[0]);
    i[0] = ntohl(tmp);

    // Copy the two consecutive 32bit ints into the Real
    memcpy(&x, i, sizeof(double));
#else
    is.read((char*)&x, sizeof(double));
#endif
}

inline void save(Real c, const std::string& filebase) {
    int taskid = mpirank();
    if (taskid != 0) {
        return;
    }
    std::string filename = appendSuffix(filebase, ".asc");
    std::ofstream os(filename.c_str());
    if (!os.good())
        cferror("save(Real, filebase) :  can't open file " + filename);
    os << std::setprecision(17);
    os << c << '\n';  // format can be read by matlab.
}

inline void load(Real& c, const std::string& filebase) {
    int taskid = mpirank();
    if (taskid == 0) {
        std::string filename = appendSuffix(filebase, ".asc");
        std::ifstream is(filename.c_str());
        if (!is.good()) {
            std::cerr << "load(Real, filebase) :  can't open file " + filename << std::endl;
            exit(1);
        }
        Real r = 0;
        is >> r;
        c = r;
    }
#ifdef HAVE_MPI
    MPI_Bcast(&c, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
}

inline void save(const Eigen::MatrixXd& A, const std::string& filebase) {
    int taskid = mpirank();

    if (taskid != 0) {
        return;
    }
    std::string filename = appendSuffix(filebase, ".asc");
    std::ofstream os(filename.c_str());
    os << std::setprecision(17);
    os << "% " << A.rows() << ' ' << A.cols() << '\n';
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j)
            os << A(i, j) << ' ';
        os << '\n';
    }
}

inline void save(const Eigen::VectorXd& x, const std::string& filebase) {
    int taskid = mpirank();

    if (taskid != 0) {
        return;
    }
    std::string filename = appendSuffix(filebase, ".asc");
    std::ofstream os(filename.c_str());
    os << std::setprecision(17);
    os << "% " << x.size() << '\n';
    for (int i = 0; i < x.size(); ++i)
        os << x(i) << '\n';
}

inline void save(Complex c, const std::string& filebase) {
    int taskid = mpirank();

    if (taskid != 0) {
        return;
    }
    std::string filename = appendSuffix(filebase, ".asc");
    std::ofstream os(filename.c_str());
    if (!os.good())
        cferror("save(Complex, filebase) :  can't open file " + filename);
    os << std::setprecision(17);
    os << Re(c) << ' ' << Im(c) << '\n';  // format can be read by matlab.
}

inline void save(const Eigen::VectorXcd& x, const std::string& filebase) {
    int taskid = mpirank();

    if (taskid != 0) {
        return;
    }
    std::string filename = appendSuffix(filebase, ".asc");
    std::ofstream os(filename.c_str());
    os << std::setprecision(17);
    os << "% " << x.size() << '\n';
    for (int i = 0; i < x.size(); ++i)
        os << Re(x(i)) << ' ' << Im(x(i)) << '\n';
}

inline void load(Eigen::MatrixXd& A, const std::string& filebase) {
    std::string filename = appendSuffix(filebase, ".asc");
    std::ifstream is(filename.c_str());
    int M, N;
    char c;
    is >> c;
    if (c != '%') {
        std::string message("load(Matrix&, filebase): bad header in file ");
        message += filename;
        cferror(message);
    }
    is >> M >> N;
    A.resize(M, N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            is >> A(i, j);
}

inline void load(Eigen::VectorXd& x, const std::string& filebase) {
    std::string filename = appendSuffix(filebase, ".asc");
    std::ifstream is(filename.c_str());
    int N;
    char c;
    is >> c;
    if (c != '%') {
        std::string message("load(VectorXd&, filebase): bad header in file ");
        message += filename;
        cferror(message);
    }
    is >> N;
    x.resize(N);
    for (int i = 0; i < x.size(); ++i)
        is >> x(i);
}

inline void load(Eigen::VectorXcd& x, const std::string& filebase) {
    std::string filename = appendSuffix(filebase, ".asc");
    std::ifstream is(filename.c_str());
    int N;
    char c;
    is >> c;
    if (c != '%') {
        std::string message("load(VectorXcd&, filebase): bad header in file ");
        message += filename;
        cferror(message);
    }
    is >> N;
    x.resize(N);
    Real xr, xi;
    for (int i = 0; i < x.size(); ++i) {
        is >> xr >> xi;
        x(i) = Complex(xr, xi);
    }
}

inline Real getRealfromLine(int taskid, std::ifstream& is) {
    Real val = 0;
    if (taskid == 0) {
        std::string line;
        getline(is, line, '%');
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        val = atof(line.c_str());
        getline(is, line);
    }

#ifdef HAVE_MPI
    MPI_Bcast(&val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    return val;
}

inline int getIntfromLine(int taskid, std::ifstream& is) {
    int val = 0;
    if (taskid == 0) {
        std::string line;
        getline(is, line, '%');
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        val = atoi(line.c_str());
        getline(is, line);
    }

#ifdef HAVE_MPI
    MPI_Bcast(&val, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    return val;
}

std::string getStringfromLine(int taskid, std::ifstream& is) {
    std::string val = "";
    if (taskid == 0) {
        std::string line;
        getline(is, line, '%');
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        val = line;
        getline(is, line);
    }

#ifdef HAVE_MPI
    auto length = static_cast<int>(val.size());
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    auto buffer = std::unique_ptr<char[]>(new char[length]);
    auto pnt = buffer.get();
    strcpy(pnt, val.c_str());
    MPI_Bcast(pnt, length, MPI_CHAR, 0, MPI_COMM_WORLD);
    val = std::string(pnt);
#endif
    return val;
}

inline void setToZero(Eigen::MatrixXd& A) {
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            A(i, j) = 0.0;
}

inline void setToZero(Eigen::VectorXd& x) {
    for (int i = 0; i < x.size(); ++i)
        x(i) = 0.0;
}

inline Real L2Norm(const Eigen::VectorXd& x, int cutoff) { return sqrt(L2Norm2(x, cutoff)); }

inline Real L2Norm(const Eigen::VectorXcd& x, int cutoff) { return sqrt(L2Norm2(x, cutoff)); }

inline Real L2Norm2(const Eigen::VectorXd& x, int cutoff) {
    Real sum = 0.0;
    for (int i = 0; i < x.size() - cutoff; ++i)
        sum += x(i) * x(i);
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    return sum;
}

inline Real L2Norm2(const Eigen::VectorXcd& x, int cutoff) {
    Real sum = 0.0;
    for (int i = 0; i < x.size() - cutoff; ++i)
        sum += abs2(x(i));
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    return sum;
}

inline Real L2Dist(const Eigen::VectorXd& x, const Eigen::VectorXd& y, int cutoff) {
    return sqrt(L2Dist2(x, y, cutoff));
}

inline Real L2Dist(const Eigen::VectorXcd& x, const Eigen::VectorXcd& y, int cutoff) {
    return sqrt(L2Dist2(x, y, cutoff));
}

inline Real L2Dist2(const Eigen::VectorXd& x, const Eigen::VectorXd& y, int cutoff) {
    assert(x.size() == y.size());
    Real sum = 0.0;
    for (int i = 0; i < x.size() - cutoff; ++i)
        sum += square(x(i) - y(i));
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    return sum;
}

inline Real L2Dist2(const Eigen::VectorXcd& x, const Eigen::VectorXcd& y, int cutoff) {
    assert(x.size() == y.size());

    Real sum = 0.0;
    for (int i = 0; i < x.size() - cutoff; ++i)
        sum += abs2(x(i) - y(i));
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    return sum;
}

inline Real L2IP(const Eigen::VectorXd& u, const Eigen::VectorXd& v, int cutoff) {
    if (u.size() != v.size()) {
        std::cerr << "error in L2IP(VectorXd, VectorXd) : vector length mismatch" << std::endl;
        exit(1);
    }
    Real sum = 0.0;
    for (int i = 0; i < u.size() - cutoff; ++i)
        sum += u(i) * v(i);
#ifdef HAVE_MPI
    Real tmp = sum;
    MPI_Allreduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    return sum;
}

inline void operator*=(Eigen::VectorXcd& x, Complex c) {
    for (int i = 0; i < x.size(); ++i)
        x(i) = x(i) * c;
}

inline void rename(const std::string& oldname, const std::string& newname) {
    if (mpirank() == 0)
        ::rename(oldname.c_str(), newname.c_str());
}

inline Real linearInterpolate(Real x0, Real y0, Real x1, Real y1, Real x) {
    return (y0 * (x - x1) - y1 * (x - x0)) / (x0 - x1);
}

inline Real linear_extrapolate(Real x0, Real f0, Real x1, Real f1, Real x) {
    return (f0 * (x - x1) - f1 * (x - x0)) / (x0 - x1);
}

// Neville algorithm, modeled after GPL v2 code from Google
inline Real polynomialInterpolate(const cfarray<Real>& fn, const cfarray<Real>& xn, Real x) {
    assert(fn.length() == xn.length());
    assert(fn.length() > 0);

    const int N = xn.length();
    const int Nbuffer = 10;
    if (N > Nbuffer) {
        cferror(
            "error in polynomial Interpolate(cfarray<Real> fn, cfarray<Real> xn, Real x) :\n\
            fn.length() == " +
            i2s(N) + " > " + i2s(Nbuffer) +
            " ==  static buffer size\n\
            this function could be extended to use dynamic memory for such cases");
    }

    if (isconst(fn))
        return fn[0];

    // If N is small, use a preallocated static buffer, else use dynamic memory
    Real y[2][Nbuffer];

    // Load input f data into y cfarray
    for (int i = 0; i < N; ++i)
        y[0][i] = fn[i];

    int j_curr = 0;
    for (int i = 1; i < N; i++) {
        j_curr = i % 2;
        int j_prev = (i - 1) % 2;
        for (int k = 0; k < (N - i); k++)
            y[j_curr][k] = linearInterpolate(xn[k], y[j_prev][k], xn[k + i], y[j_prev][k + 1], x);
    }
    return y[j_curr][0];
}

// Neville algorithm, modled after GPL v2 code from Google
inline Real quadraticInterpolate(const cfarray<Real>& fn, const cfarray<Real>& xn, Real x) {
    assert(fn.N() >= 3);
    assert(xn.N() >= 3);
    if (fn[0] == fn[1] && fn[1] == fn[2])
        return fn[1];

    const int N = 3;  // three points for quadratic extrapolation
    Real y[2][N];     // two cfarrays for quadratic extrap recurrence relation

    // Load input f data into y cfarray
    for (int i = 0; i < N; ++i)
        y[0][i] = fn[i];

    int j_curr = 0;
    for (int i = 1; i < N; i++) {
        j_curr = i % 2;
        int j_prev = (i - 1) % 2;
        for (int k = 0; k < (N - i); k++)
            y[j_curr][k] = linearInterpolate(xn[k], y[j_prev][k], xn[k + i], y[j_prev][k + 1], x);
    }
    return y[j_curr][0];
}

// Neville algorithm.
inline Real quadratic_extrapolate(const cfarray<Real>& f, const cfarray<Real>& mu, Real mug) {
    const int N = 3;  // three points for quadratic extrapolation
    Real y[2][N];     // two cfarrays for quadratic extrap recurrence relation

    // Load input f data into y cfarray
    for (int i = 0; i < N; ++i)
        y[0][i] = f[i];

    int j_curr;
    for (int i = 1; i < N; i++) {
        j_curr = i % 2;
        int j_prev = (i - 1) % 2;
        for (int k = 0; k < (N - i); k++)
            y[j_curr][k] = linear_extrapolate(mu[k], y[j_prev][k], mu[k + i], y[j_prev][k + 1], mug);
    }
    return y[j_curr][0];
}

/// Get the filename without path and without extension
inline std::string getfilebase(std::string filename) {
    // remove path (if any)
    size_t pos = filename.rfind("/");
    if (pos != std::string::npos)
        filename = filename.substr(pos + 1);

    // remove extension (if any)
    pos = filename.rfind(".");
    if (pos != std::string::npos)
        return filename.substr(0, pos);
    return filename;
}

inline std::string getfileextension(std::string filename) {
    // remove path (if any)
    size_t pos = filename.rfind("/");
    if (pos != std::string::npos)
        filename = filename.substr(pos + 1);

    // get extension
    pos = filename.rfind(".");
    if (pos != std::string::npos)
        return filename.substr(pos + 1);
    return "";
}

inline bool checkFlagContent(std::ifstream& is, const std::vector<std::string>& flagList) {
    std::string tmp;
    for (uint i = 0; i < flagList.size(); i++) {
        is >> tmp >> tmp;
        if (tmp != flagList[i]) {
            is.clear();
            is.seekg(0, std::ios::beg);
            return false;
        }
    }
    is.clear();
    is.seekg(0, std::ios::beg);
    return true;
}

inline std::ostream& operator<<(std::ostream& os, HookstepPhase p) {
    if (p == ReducingDelta)
        os << "ReducingDelta";
    else if (p == IncreasingDelta)
        os << "IncreasingDelta";
    else
        os << "Finished";
    return os;
}
inline std::ostream& operator<<(std::ostream& os, ResidualImprovement i) {
    if (i == Unacceptable)
        os << "Unacceptable";
    else if (i == Poor)
        os << "Poor";
    else if (i == Ok)
        os << "Ok";
    else if (i == Good)
        os << "Good";
    else if (i == Accurate)
        os << "Accurate";
    else
        os << "NegativeCurvature";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, SolutionType solntype) {
    os << (solntype == Equilibrium ? "Equilibrium" : "PeriodicOrbit");
    return os;
}

inline void rescale(Eigen::VectorXd& x, const Eigen::VectorXd& xscale) {
    if (xscale.size() != 0)
        for (int i = 0; i < x.size(); ++i)
            x(i) /= xscale(i);
}
inline void unscale(Eigen::VectorXd& x, const Eigen::VectorXd& xscale) {
    if (xscale.size() != 0)
        for (int i = 0; i < x.size(); ++i)
            x(i) *= xscale(i);
}

inline Real adjustDelta(Real delta, Real deltaRate, Real deltaMin, Real deltaMax, std::ostream& os) {
    os << "  old delta == " << delta << std::endl;

    // A special case: if deltaRate would take us from above to below deltaMin,
    // set it at exactly deltaMin and give it one last try. If next time throgh
    // we try to go even lower, go ahead and set it lower, and the calling function
    // will check and return best current answer

    if (delta == deltaMin && delta * deltaRate < deltaMin) {
        os << "delta bottoming out at deltaMin, try one more search" << std::endl;
        return deltaMin;
    }

    delta *= deltaRate;
    if (delta <= deltaMin) {
        delta = 0.5 * deltaMin;
        os << "delta bottomed out at deltaMin" << std::endl;
    }
    if (delta > deltaMax) {
        delta = deltaMax;
        os << "delta topped out at deltaMax" << std::endl;
    }
    os << "  new delta == " + r2s(delta) << std::endl;
    return delta;
}

inline Real adjustLambda(Real lambda, Real lambdaMin, Real lambdaMax, std::ostream& os) {
    if (lambda < lambdaMin) {
        os << "lambda == " + r2s(lambda) + " is too small. Resetting to lambda == " << lambdaMin << std::endl;
        lambda = lambdaMin;
    } else if (lambda < lambdaMin) {
        os << "lambda == " + r2s(lambda) + " is too large. Resetting to lambda == " << lambdaMax << std::endl;
        lambda = lambdaMax;
    }
    return lambda;
}

inline void solve(const Eigen::MatrixXd& Ut, const Eigen::VectorXd& D, const Eigen::MatrixXd& V,
                  const Eigen::VectorXd& b, Eigen::VectorXd& x) {
    Eigen::VectorXd bh = Ut * b;
    const Real eps = 1e-12;
    for (int i = 0; i < D.size(); ++i) {
        if (std::abs(D(i)) > eps)
            bh(i) *= 1.0 / D(i);
        else
            bh(i) = 0.0;
    }
    x = V * bh;
}

// returns u dot v/(|u||v|) or zero if either has zero norm
inline Real align(const Eigen::VectorXd& u, const Eigen::VectorXd& v) {
    Real norm = L2Norm(u) * L2Norm(v);
    if (norm == 0.0)
        return 0.0;
    else
        return L2IP(u, v) / norm;
}

inline Real executionTime() {
    static bool firstCall = true;
    static timeval start;
    if (firstCall) {
        gettimeofday(&start, 0);
        firstCall = false;
        return 0;
    }

    timeval end;

    gettimeofday(&end, 0);
    Real sec = (Real)(end.tv_sec - start.tv_sec);
    Real ms = (((Real)end.tv_usec) - ((Real)start.tv_usec));
    Real exectime = sec + ms / 1000000.;
    return exectime;
}

// Algorithm from Numerical Recipes in C pg 110;
inline Real polyInterp(const cfarray<Real>& fa, const cfarray<Real>& xa, Real x) {
    assert(fa.N() == xa.N());
    int N = fa.N();
    cfarray<Real> C(N);
    cfarray<Real> D(N);

    Real xdiff = std::abs(x - xa[0]);
    Real xdiff2 = 0.0;
    Real df;

    int i_closest = 0;
    for (int i = 0; i < N; ++i) {
        if ((xdiff2 = std::abs(x - xa[i])) < xdiff) {
            i_closest = i;
            xdiff = xdiff2;
        }
        C[i] = fa[i];
        D[i] = fa[i];
    }

    // Initial approx to f(x)
    Real fx = fa[i_closest--];

    for (int m = 1; m < N; ++m) {
        for (int i = 0; i < N - m; ++i) {
            Real C_dx = xa[i] - x;
            Real D_dx = xa[i + m] - x;
            Real C_D = C[i + 1] - D[i];
            Real denom = C_dx - D_dx;
            if (denom == 0.0)
                cferror("polyinterp(fa,xa,x) : two values of xa has are equal!");
            denom = C_D / denom;
            C[i] = C_dx * denom;
            D[i] = D_dx * denom;
        }
        fx += (df = (2 * (i_closest + 1) < (N - m) ? C[i_closest + 1] : D[i_closest--]));
    }
    return fx;
}
inline Real secantSearch(Real a, Real b, cfarray<Real>& fn, const cfarray<Real>& xn, Real feps, int maxsteps) {
    Real fa = polyInterp(fn, xn, a);
    Real fb = polyInterp(fn, xn, b);
    if (fa * fb > 0)
        cferror("secantSearch(a,b,fn,xn) : a and b don't bracket a zero");

    Real c = a - fa * (b - a) / (fb - fa);
    Real fc;

    for (int n = 0; n < maxsteps; ++n) {
        c = a - fa * (b - a) / (fb - fa);
        fc = polyInterp(fn, xn, c);
        if (std::abs(fc) < feps)
            break;
        if (fc * fa > 0) {
            a = c;
            fa = fc;
        } else {
            b = c;
            fb = fc;
        }
    }
    return c;
}

inline Real bisectSearch(Real a, Real b, cfarray<Real>& fn, const cfarray<Real>& xn, Real feps, int maxsteps) {
    Real c = 0.5 * (a + b) / 2;  // GCC-3.3.5 complains if not explicitly init'ed
    Real fc;
    Real fa = polyInterp(fn, xn, a);
    Real fb = polyInterp(fn, xn, b);
    if (fa * fb > 0)
        cferror("bisectSearch(a,b,fn,xn) : a and b don't bracket a zero");

    for (int n = 0; n < maxsteps; ++n) {
        c = 0.5 * (a + b);
        fc = polyInterp(fn, xn, c);
        if (std::abs(fc) < feps)
            break;
        if (fc * fa > 0) {
            a = c;
            fa = fc;
        } else {
            b = c;
            fb = fc;
        }
    }
    return c;
}

// If s is numeric, convert to real using atof

inline std::string FillSpaces(int i, int n) {
    std::string s = i2s(i);
    int l = s.length();
    for (int j = l; j < n; j++) {
        s = " " + s;
    }
    return s;
}
inline std::string FillSpaces(Real t, int n) { return FillSpaces((int)t, n); }

inline bool isconst(cfarray<Real> f, Real eps) {
    if (f.length() == 0)
        return true;

    Real f0 = f[0];
    bool rtn = true;
    for (int n = 0; n < f.length(); ++n)
        if (std::abs(f[n] - f0) > eps) {
            rtn = false;
            break;
        }
    return rtn;
}

inline bool fileExists(const std::string& filename) {
    int res = 0;
    int taskid = mpirank();
    if (taskid == 0) {
        struct stat st;
        res = (stat(filename.c_str(), &st) == 0) ? 1 : 0;
    }

#ifdef HAVE_MPI
    MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    return (res == 1);
}

inline std::string FillZeros(int i, int n) {
    std::string s = i2s(i);
    int l = s.length();
    for (int j = l; j < n; j++) {
        s = "0" + s;
    }
    return s;
}

inline std::string FillZeros(Real t, int n) { return FillZeros((int)t, n); }

inline void mkdir(const std::string& dirname) {
    if (mpirank() == 0)
        ::mkdir(dirname.c_str(), S_IRUSR | S_IWUSR | S_IXUSR);
}

inline std::string pwd() {
    const int N = 512;
    char* buff = new char[N];
    if (getcwd(buff, N) == NULL)
        cferror("Error in getcwd()");
    std::string rtn(buff);
    delete[] buff;
    return rtn;
}

inline std::string pathfix(const std::string& path) {
    std::string rtn = path;
    if (rtn.length() > 0 && rtn[rtn.length() - 1] != '/')
        rtn += "/";
    return rtn;
}

inline std::string stub(const std::string& filename, const std::string& ext) {
    // Clip off path
    size_t s = filename.find_last_of("/");
    std::string f = (s == std::string::npos) ? filename : filename.substr(s + 1, filename.length() - s);

    // Clip off extension
    size_t t = f.find(ext, f.length() - ext.length());
    std::string g = (t == std::string::npos) ? f : f.substr(0, t);

    return g;
}

inline std::string t2s(Real t, bool inttime) {
    char buffer[12];
    if (inttime) {
        int it = iround(t);
        sprintf(buffer, "%d", it);
    } else
        sprintf(buffer, "%.3f", t);
    return std::string(buffer);
}

inline std::string fuzzyless(Real x, Real eps) {
    std::string rtn("false ");
    if (x < eps)
        rtn = "TRUE  ";
    else if (x < sqrt(eps))
        rtn = "APPROX";
    return rtn;
}

template <class T>
inline void push(const T& t, cfarray<T>& a) {
    for (int n = a.length() - 1; n > 0; --n)
        a[n] = a[n - 1];
    a[0] = t;
}

inline void openfile(std::ofstream& f, std::string filename, std::ios::openmode openflag) {
    int rank = mpirank();
    if (rank == 0) {
        f.open(filename.c_str(), openflag);
    } else {
        f.open("/dev/null");
    }
}

/**
 * @brief Sort the vector of eigenvalues passed in input by the abs of its values
 * in decreasing order and permute the matrix of eigenvectors accordingly.
 *
 * @tparam Vector type of the input vector
 * @tparam Matrix type of the input matrix
 *
 * @param[in,out] Lambda vector of eigenvalues
 * @param[in,out] V matrix of the corresponding eigenvectors
 */
template <class Vector, class Matrix>
inline void sort_by_abs(Vector& Lambda, Matrix& V) {
    assert(Lambda.size() == V.cols());

    // Get the order of the elements in Lambda by magnitude.
    Eigen::VectorXi indexes(V.cols());

    std::iota(indexes.data(), indexes.data() + indexes.size(), 0);
    std::sort(indexes.data(), indexes.data() + indexes.size(),
              [&Lambda](const size_t& ii, const size_t& jj) { return std::abs(Lambda(ii)) > std::abs(Lambda(jj)); });

    // Construct a permutation matrix from those indexes and permute both
    // Lambda and V
    Eigen::PermutationWrapper<Eigen::VectorXi> P(indexes);

    V = V * P;
    Lambda = P.transpose() * Lambda;
}

}  // namespace chflow

#endif  // CFBASICS
