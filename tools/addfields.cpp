/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */
#include <strings.h>
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
        string purpose("create a linear combination of flowfields (-lc) or add baseflow to a given flowfield (-ab)\n");
        ArgList args(argc, argv, purpose);

        const bool ab = args.getflag("-ab", "--addbaseflow", "add baseflow to the input flowfield");
        const bool lc = args.getflag("-lc", "--linearcombination",
                                     "create a linear combination of flowfields"
                                     "\n\t\t\t\t\t\t\t\t   usage : addfields.x  c0 u0  c1 u1 c2 u2 ... <outfield>"
                                     "\n\t\t\t\t\t\t\t\t   where cn are real constants and un are flowfields\n");
        string outname = args.getstr(1, "<outfield>", "filename for output field");

        string Uname, Wname;
        DNSFlags baseflags = setBaseFlowFlags(args, Uname, Wname);
        const Real ubasefac = args.getreal("-Uf", "--Uf", 1, "Multiply baseflow by this factor before adding");

        if (ab & !lc) {
            const string uname = args.getstr(2, "<infield>", "input flowfield");
            FlowField u(uname);
            u.makeSpectral();

            cout << "Creating base flow:" << endl;
            vector<ChebyCoeff> base_Flow = baseFlow(u.Ny(), u.a(), u.b(), baseflags, Uname, Wname);
            for (int i = 0; i < 2; i++) {
                base_Flow[i].makePhysical();
                base_Flow[i] *= ubasefac;
                base_Flow[i].makeSpectral();
            }
            u += base_Flow;
            u.cmplx(0, 0, 0, 1) -= Complex(ubasefac * baseflags.Vsuck, 0.);

            args.check();
            args.save("./");
            u.save(outname);
        }

        else if (lc & !ab) {
            if (argc % 2 != 0 && argc < 4) {
                cferror("please use at least one real/field pair and an output file as command-line arguments\n");
            }

            FlowField sum;

            // Minimum command line: argv0 c0 u0 uname
            //	                        3   2    1

            for (int n = 2; n < argc - 1; n += 2) {
                string uname = args.getstr(n, "<field>", "flowfield");
                Real c = args.getreal(n + 1, "<Real>", "coefficient");

                FlowField u(uname);
                u *= c;

                if (n == 2) {
                    cout << outname << "  = " << c << " * " << uname << endl;
                    sum = u;
                } else {
                    if (!sum.congruent(u)) {
                        cerr << "FlowField " << uname << " is not compatible with previous fields.\n" << endl;
                        exit(1);
                    }

                    cout << outname << " += " << c << " * " << uname << endl;
                    sum += u;
                }
            }

            args.check();
            args.save("./");
            sum.save(outname);
        }

        else {
            args.check();
            args.save("./");
            cferror("Please enter one of the two possible options: -ab (add base flow) or -lc (linear combination)");
        }
    }
    cfMPI_Finalize();
}
