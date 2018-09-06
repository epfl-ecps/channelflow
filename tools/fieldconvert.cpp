/**
 * Program for writing HDF5 files from flowfield data, essentially a
 * modification of field2ascii. Some of the code and the accopmanying
 * comments are taken from the HDF group website (www.hdfgroup.org).
 *
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include "channelflow/dns.h"
#include "channelflow/flowfield.h"
#include "channelflow/utilfuncs.h"

using namespace std;
using namespace chflow;

int main(int argc, char* argv[]) {
    cfMPI_Init(&argc, &argv);
    {
        string purpose(
            "conversion options:\n"
            " \t.ff(binary) <---> .h5(HDF5)    .ff(binary)<--->.nc(NetCDF)     .h5(HDF5)<--->.nc(NetCDF)    in both "
            "directions\n"
            " \tFlowField(ff,hdf5,nc)--->.asc (asci)\n"
            " \tFlowField(ff,hdf5,nc)--->.vtk \n");

        ArgList args(argc, argv, purpose);
        const string iname = args.getstr(2, "<filename>", "input field");
        const string oname = args.getstr(1, "<filename>", "output field");

        args.check();

        FlowField u(iname);
        u.save(oname);
    }
    cfMPI_Finalize();
}
