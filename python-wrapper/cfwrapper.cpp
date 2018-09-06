/**
 * This file is a part of channelflow version 2.0 https://channelflow.ch.
 * License is GNU GPL version 2 or later: ./LICENCE
 */

#include <boost/python.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "channelflow/diffops.h"
#include "channelflow/flowfield.h"
#include "channelflow/symmetry.h"
#include "modules/viscoelastic/veutils.h"

using namespace std;
using namespace boost::python;

namespace chflow {

/*** Access functions for FlowField values ***/
Real FlowField_get_val(FlowField& self, boost::python::tuple t) {
    extract<int> nx(t[0]), ny(t[1]), nz(t[2]), i(t[3]);  // A check that len(t) = 4 would be helpful
    if (self.xzstate() == Spectral)
        throw runtime_error("Trying to access physical data, but FlowField is in spectral state");
    return self(nx, ny, nz, i);
}

void FlowField_set_val(FlowField& self, boost::python::tuple t, Real val) {
    if (self.xzstate() == Spectral)
        throw runtime_error("Trying to access physical data, but FlowField is in spectral state");
    extract<int> nx(t[0]), ny(t[1]), nz(t[2]), i(t[3]);
    self(nx, ny, nz, i) = val;
}

Complex FlowField_get_cmplx(FlowField& self, int nx, int ny, int nz, int i) {
    if (self.xzstate() == Physical)
        throw runtime_error("Trying to access spectral data, but FlowField is in physical state");
    return self.cmplx(nx, ny, nz, i);
}

/*** Wrapper functions ***/
// They are needed because Python can't handle overloaded functions. So for each
// function with a default argument, we need a wrapper function that has a fixed
// number of arguments
Real L2Norm_wrapped(const FlowField& f) { return L2Norm(f); }
Real L2Norm2_wrapped(const FlowField& f) { return L2Norm2(f); }
Real L2Dist_wrapped(const FlowField& f, const FlowField& g) { return L2Dist(f, g); }
Real L2IP_wrapped(const FlowField& f, const FlowField& g) { return L2InnerProduct(f, g); }
Real wallshear_wrapped(const FlowField& f) { return wallshear(f); }
Real wallshearUpper_wrapped(const FlowField& f) { return wallshearUpper(f); }
Real wallshearLower_wrapped(const FlowField& f) { return wallshearLower(f); }
Real L2Norm3d_wrapped(const FlowField& f) { return L2Norm3d(f); }
Real Ecf_wrapped(const FlowField& f) { return Ecf(f); }
FlowField curl_wrapped(const FlowField& f) { return curl(f); }
FlowField lapl_wrapped(const FlowField& f) { return lapl(f); }
FlowField grad_wrapped(const FlowField& f) { return grad(f); }
FlowField div_wrapped(const FlowField& f) { return div(f); }

FlowField diff_wrapped(const FlowField& f, int i, int n) { return diff(f, i, n); }
void FlowField_save(const FlowField& self, string filebase) { self.save(filebase); }

/*** The actual python module ***/
BOOST_PYTHON_MODULE(libpycf) {
    class_<FlowField>("FlowField", init<>())
        .def(init<string>())
        .def(init<FlowField>())
        .def(init<int, int, int, int, Real, Real, Real, Real>())
        .def("save", &FlowField_save)
        .def("get", &FlowField_get_val)
        .def("cmplx", &FlowField_get_cmplx)
        .def("__getitem__", &FlowField_get_val)
        .def("__setitem__", &FlowField_set_val)
        .def("makePhysical", &FlowField::makePhysical)
        .def("makeSpectral", &FlowField::makeSpectral)
        .def("setToZero", &FlowField::setToZero)
        .add_property("Nx", &FlowField::Nx)
        .add_property("Ny", &FlowField::Ny)
        .add_property("Nz", &FlowField::Nz)
        .add_property("Nd", &FlowField::Nd)
        .add_property("Mx", &FlowField::Mx)
        .add_property("My", &FlowField::My)
        .add_property("Mz", &FlowField::Mz)
        .add_property("Lx", &FlowField::Lx)
        .add_property("Ly", &FlowField::Ly)
        .add_property("Lz", &FlowField::Lz)
        .add_property("a", &FlowField::a)
        .add_property("b", &FlowField::b)
        .def("x", &FlowField::x)
        .def("y", &FlowField::y)
        .def("z", &FlowField::z)
        .def(self *= Real())
        .def(self += self)
        .def(self -= self);

    class_<FieldSymmetry>("FieldSymmetry", init<>())
        .def(init<FieldSymmetry>())
        .def(init<string>())
        .def(init<int, int, int, Real, Real, int>())
        .add_property("sx", &FieldSymmetry::sx)
        .add_property("sy", &FieldSymmetry::sy)
        .add_property("sz", &FieldSymmetry::sz)
        .add_property("ax", &FieldSymmetry::ax)
        .add_property("az", &FieldSymmetry::az)
        .def("__call__",
             static_cast<FlowField (FieldSymmetry::*)(const FlowField& u) const>(&FieldSymmetry::operator()));

    def("L2Norm", &L2Norm_wrapped);
    def("L2Norm2", &L2Norm2_wrapped);
    def("L2Dist", &L2Dist_wrapped);
    def("L2IP", &L2IP_wrapped);
    def("wallshear", &wallshear_wrapped);
    def("wallshearLower", &wallshearLower_wrapped);
    def("wallshearUpper", &wallshearUpper_wrapped);
    def("L2Norm3d", &L2Norm3d_wrapped);
    def("curl", &curl_wrapped);
}

}  // namespace chflow
