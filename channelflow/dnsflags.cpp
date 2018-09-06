/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 */

#include "channelflow/dnsflags.h"

#include <type_traits>

using namespace std;

namespace {
/**
 * Parse an enum value from an input stream line
 *
 * @param[in,out] is input stream from which the line is read
 * @param[in] parse function-like object that parse a line and contructs
 *     a value
 * @param[in] default_value default value used to initialize the output
 *
 * @returns The parsed enum value
 */
template <class T, class ParseFn>
T parse_enum_value(std::istream& is, ParseFn parse, T default_value) {
    static_assert(std::is_enum<T>::value, "The default value must be of an enum type");

    auto val = default_value;
    auto mpi_rank = 0;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
    if (mpi_rank == 0) {
        string line;
        getline(is, line, '%');
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        val = parse(line);
        getline(is, line);
    }

#ifdef HAVE_MPI
    // To make this function work regardless of the type underlying the enum
    // we copy val into a large enough buffer, broadcast the buffer and then
    // copy back
    auto buffer = static_cast<int64_t>(val);
    MPI_Bcast(&buffer, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    val = static_cast<T>(buffer);
#endif
    return val;
}
}  // namespace

namespace chflow {

int getBodyforcefromLine(int taskid, std::ifstream& is) {
    int val = 0;
    if (taskid == 0) {
        string line;
        getline(is, line, '%');
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (line == "zero_bodyforce")
            val = 0;
        else
            cferror("bodyforce not zero, this is not implemented for a restart.");
        getline(is, line);
    }

#ifdef HAVE_MPI
    MPI_Bcast(&val, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    return val;
}

BodyForce::BodyForce() {}

// virtual
Vector BodyForce::operator()(Real x, Real y, Real z, Real t) {
    Vector f(3);
    this->eval(x, y, z, t, f(0), f(1), f(2));
    return f;
}

void BodyForce::eval(Real x, Real y, Real z, Real t, Real& fx, Real& fy, Real& fz) {
    fx = 0.0;
    fy = 0.0;
    fz = 0.0;
}

// virtual
void BodyForce::eval(Real t, FlowField& f) {
    f.makeState(Physical, Physical);
    f.setToZero();
    assert(f.Nd() == 3);
    Vector v;
    for (int ny = 0; ny < f.Ny(); ++ny) {
        Real y = f.y(ny);
        for (int nx = 0; nx < f.Nx(); ++nx) {
            Real x = f.x(nx);
            for (int nz = 0; nz < f.Nz(); ++nz) {
                Real z = f.z(nz);
                this->eval(x, y, z, t, f(nx, ny, nz, 0), f(nx, ny, nz, 1), f(nx, ny, nz, 2));
            }
        }
    }
    f.makeSpectral();
}

// virtual
bool BodyForce::isOn(Real t) { return true; }

DNSFlags::DNSFlags(Real nu_, Real dPdx_, Real dPdz_, Real Ubulk_, Real Wbulk_, Real Uwall_, Real ulowerwall_,
                   Real uupperwall_, Real wlowerwall_, Real wupperwall_, Real theta_, Real Vsuck_, Real rotation_,
                   Real t0_, Real T_, Real dT_, Real dt_, bool variabledt_, Real dtmin_, Real dtmax_, Real CFLmin_,
                   Real CFLmax_, Real symmetryprojectioninterval_, BaseFlow baseflow_, MeanConstraint constraint_,
                   TimeStepMethod timestepping_, TimeStepMethod initstepping_, NonlinearMethod nonlinearity_,
                   Dealiasing dealiasing_, BodyForce* bodyforce_, bool taucorrection_, Verbosity verbosity_,
                   ostream* logstream_)
    : baseflow(baseflow_),
      constraint(constraint_),
      timestepping(timestepping_),
      initstepping(initstepping_),
      nonlinearity(nonlinearity_),
      dealiasing(dealiasing_),
      bodyforce(bodyforce_),
      taucorrection(taucorrection_),
      nu(nu_),
      Vsuck(Vsuck_),
      rotation(rotation_),
      theta(theta_),
      dPdx(dPdx_),
      dPdz(dPdz_),
      Ubulk(Ubulk_),
      Wbulk(Wbulk_),
      Uwall(Uwall_),
      ulowerwall(ulowerwall_),
      uupperwall(uupperwall_),
      wlowerwall(wlowerwall_),
      wupperwall(wupperwall_),
      t0(t0_),
      T(T_),
      dT(dT_),
      dt(dt_),
      variabledt(variabledt_),
      dtmin(dtmin_),
      dtmax(dtmax_),
      CFLmin(CFLmin_),
      CFLmax(CFLmax_),
      symmetryprojectioninterval(symmetryprojectioninterval_),
      verbosity(verbosity_),
      logstream(logstream_) {
    symmetries = SymmetryList();
    if (dealias_y() && (nonlinearity != Rotational)) {
        cerr << "DNSFlags::DNSFlags: DealiasY and DealiasXYZ work only with\n";
        cerr << "Rotational nonlinearity in the current version of channelflow.\n";
        cerr << "Setting nonlinearity to Rotational." << endl;
        nonlinearity = Rotational;
    }
    // maybe should print warnings about initstepping only mattering for SBDF and CNAB
}

DNSFlags::DNSFlags(ArgList& args, const bool laurette)
    :

      DNSFlags()  // Default constructor called to initialize some boolean/pointers (e.g. bodyforce ... )
{
    // dimensionless parameters of the system
    args.section("System parameters");
    const Real Reynolds = args.getreal("-R", "--Reynolds", 400, "pseudo-Reynolds number == 1/nu");
    const Real nuarg =
        args.getreal("-nu", "--nu", 0, "kinematic viscosity (takes precedence over Reynolds, if nonzero)");
    nu = (nuarg != 0) ? nuarg : 1.0 / Reynolds;
    // more general dnsflags
    args2BC(args);
    args2numerics(args, laurette);
}

void DNSFlags::args2BC(ArgList& args) {
    // boundary conditions
    args.section("Boundary conditions");
    const string basestr_ = args.getstr("-bf", "--baseflow", "laminar",
                                        "set base flow to one of [zero|laminar|linear|parabolic|suction|arbitrary]");
    const string meanstr_ =
        args.getstr("-mc", "--meanconstraint", "gradp", "fix one of two flow constraints [gradp|bulkv]");
    const Real dPds_ =
        args.getreal("-dPds", "--dPds", 0.0, "magnitude of imposed pressure gradient along streamwise s");
    const Real Ubulk_ = args.getreal("-Ubulk", "--Ubulk", 0.0, "magnitude of imposed bulk velocity");
    const Real Uwall_ =
        args.getreal("-Uwall", "--Uwall", 1.0, "magnitude of imposed wall velocity, +/-Uwall at y = +/-h");
    const Real theta_ = args.getreal("-theta", "--theta", 0.0, "angle of base flow relative to x-axis");
    const Real Vsuck_ = args.getreal("-Vs", "--Vsuck", 0.0, "wall-normal suction velocity");
    const Real rotation_ = args.getreal("-rot", "--rotation", 0.0, "rotation around the z-axis");

    baseflow = s2baseflow(basestr_);
    constraint = s2constraint(meanstr_);
    theta = theta_;
    Uwall = Uwall_;
    ulowerwall = -Uwall_ * cos(theta_);
    uupperwall = Uwall_ * cos(theta_);
    wlowerwall = -Uwall_ * sin(theta_);
    wupperwall = Uwall_ * sin(theta_);
    Vsuck = Vsuck_;
    rotation = rotation_;
    dPdx = dPds_ * cos(theta_);
    dPdz = dPds_ * sin(theta_);
    Ubulk = Ubulk_ * cos(theta_);
    Wbulk = Ubulk_ * sin(theta_);
}

void DNSFlags::args2numerics(ArgList& args, const bool laurette) {
    // numerical setup
    args.section("Numerical setup");
    const string dealiasstr_ =
        args.getstr("-da", "--dealiasing", "DealiasXZ",
                    "define dealiasing behavior, one of [NoDealiasing|DealiasXZ|DealiasY|DealiasXYZ]");
    const Real T0_ = args.getreal("-T0", "--T0", 0.0, "start time of DNS or period of map f^T(u)");
    const Real T_ = args.getreal("-T", "--T1", 20.0, "final time of DNS or period of map f^T(u)");
    const Real dT_ = args.getreal("-dT", "--dT", 1.0, "save interval");
    const Real dtarg_ = args.getreal("-dt", "--dt", 0.03125, "timestep");
    const bool vardt_ = args.getbool("-vdt", "--variabledt", true, "adjust dt to keep CFLmin<=CFL<CFLmax");
    const Real dtmin_ = args.getreal("-dtmin", "--dtmin", 0.001, "minimum time step");
    const Real dtmax_ = args.getreal("-dtmax", "--dtmax", 0.2, "maximum time step");
    const Real CFLmin_ = args.getreal("-CFLmin", "--CFLmin", 0.40, "minimum CFL number");
    const Real CFLmax_ = args.getreal("-CFLmax", "--CFLmax", 0.60, "maximum CFL number");
    const string stepstr_ = args.getstr("-ts", "--timestepping", "sbdf3",
                                        "timestepping algorithm, "
                                        " one of [cnfe1|cnab2|cnrk2|smrk2|sbdf1|sbdf2|sbdf3|sbdf4]");
    const string initstr_ = args.getstr("-is", "--initstepping", "smrk2",
                                        "timestepping algorithm for initializing multistep algorithms, "
                                        " one of [cnfe1|cnrk2|smrk2|sbdf1]");
    const string nonlstr_ = args.getstr("-nl", "--nonlinearity", "rot",
                                        "method of calculating "
                                        "nonlinearity, one of [rot|conv|div|skew|alt|linear]");
    const int symmpi_ =
        args.getint("-symmpi", "--symmetryprojection", 100, "project onto symmetries at this time interval");
    const string symmstr_ = args.getstr("-symms", "--symmetries", "",
                                        "constrain u(t) to invariant "
                                        "symmetric subspace, argument is the filename for a file "
                                        "listing the generators of the isotropy group");

    initstepping = s2stepmethod(initstr_);
    timestepping = s2stepmethod(stepstr_);
    nonlinearity = s2nonlmethod(nonlstr_);
    dealiasing = s2dealiasing(dealiasstr_);  // DealiasXZ;
    t0 = T0_;
    T = T_;
    dT = dT_;
    dt = dtarg_;
    variabledt = vardt_;
    dtmin = dtmin_;
    dtmax = dtmax_;
    CFLmin = CFLmin_;
    CFLmax = CFLmax_;
    symmetryprojectioninterval = symmpi_;
    verbosity = Silent;
    if (symmstr_.length() > 0) {
        SymmetryList symms(symmstr_);
        symmetries = symms;
    }

    if (laurette) {  // AY MF
        dT = T;
        dt = T;
        dtmax = T;
        variabledt = false;
        initstepping = SBDF1;
        timestepping = SBDF1;
    }
}

bool DNSFlags::dealias_xz() const { return ((dealiasing == DealiasXZ || dealiasing == DealiasXYZ) ? true : false); }

bool DNSFlags::dealias_y() const { return ((dealiasing == DealiasY || dealiasing == DealiasXYZ) ? true : false); }

ostream& operator<<(ostream& os, VelocityScale v) { return os << velocityscale2string(v); }
ostream& operator<<(ostream& os, BaseFlow b) { return os << baseflow2string(b); }
ostream& operator<<(ostream& os, MeanConstraint m) { return os << constraint2string(m); }
ostream& operator<<(ostream& os, TimeStepMethod t) { return os << stepmethod2string(t); }
ostream& operator<<(ostream& os, NonlinearMethod nonl) { return os << nonlmethod2string(nonl); }
ostream& operator<<(ostream& os, Dealiasing d) { return os << dealiasing2string(d); }
ostream& operator<<(ostream& os, Verbosity v) { return os << verbosity2string(v); }

string dealiasing2string(Dealiasing d) {
    string s;
    switch (d) {
        case NoDealiasing:
            s = "NoDealiasing";
            break;
        case DealiasXZ:
            s = "DealiasXZ";
            break;
        case DealiasY:
            s = "DealiasY";
            break;
        case DealiasXYZ:
            s = "DealiasXYZ";
            break;
        default:
            s = "Invalid Dealiasing value: please submit bug report";
    }
    return s;
}

string constraint2string(MeanConstraint m) {
    string s;
    switch (m) {
        case PressureGradient:
            s = "PressureGradient";
            break;
        case BulkVelocity:
            s = "BulkVelocity";
            break;
        default:
            s = "Invalid MeanConstraint value: please submit bug report";
    }
    return s;
}

string stepmethod2string(TimeStepMethod t) {
    string s;
    switch (t) {
        case CNFE1:
            s = "CNFE1";
            break;
        case CNAB2:
            s = "CNAB2";
            break;
        case CNRK2:
            s = "CNRK2";
            break;
        case SMRK2:
            s = "SMRK2";
            break;
        case SBDF1:
            s = "SBDF1";
            break;
        case SBDF2:
            s = "SBDF2";
            break;
        case SBDF3:
            s = "SBDF3";
            break;
        case SBDF4:
            s = "SBDF4";
            break;
        default:
            s = "Invalid TimeStepMethod value: please submit bug report";
    }
    return s;
}

string verbosity2string(Verbosity v) {
    string s;
    switch (v) {
        case Silent:
            s = "Silent";
            break;
        case PrintTicks:
            s = "PrintTicks";
            break;
        case PrintTime:
            s = "PrintTime";
            break;
        case VerifyTauSolve:
            s = "VerifyTauSolve";
            break;
        case PrintAll:
            s = "PrintAll";
            break;
        default:
            s = "Invalid Verbosity value: please submit bug report";
    }
    return s;
}

string velocityscale2string(VelocityScale v) {
    string s;
    switch (v) {
        case WallScale:
            s = "WallScale";
            break;
        case ParabolicScale:
            s = "ParabolicScale";
            break;
            // case BulkScale:        s="BulkScale"; break;
        default:
            s = "Invalid VelocityScale: please submit bug report";
    }
    return s;
}

string baseflow2string(BaseFlow b) {
    string s;
    switch (b) {
        case ZeroBase:
            s = "ZeroBase";
            break;
        case LinearBase:
            s = "LinearBase";
            break;
        case ParabolicBase:
            s = "ParabolicBase";
            break;
        case LaminarBase:
            s = "LaminarBase";
            break;
        case SuctionBase:
            s = "SuctionBase";
            break;
        case ArbitraryBase:
            s = "ArbitraryBase";
            break;
        default:
            s = "Invalid BaseFlow: please submit bug report";
    }
    return s;
}

string nonlmethod2string(NonlinearMethod n) {
    string s;
    switch (n) {
        case Rotational:
            s = "Rotational";
            break;
        case Convection:
            s = "Convection";
            break;
        case Divergence:
            s = "Divergence";
            break;
        case SkewSymmetric:
            s = "SkewSymmetric";
            break;
        case Alternating:
            s = "Alternating";
            break;
        case Alternating_:
            s = "Alternating_";
            break;
        case LinearAboutProfile:
            s = "LinearAboutProfile";
            break;
            // case LinearAboutField: s="LinearAboutField"; break;
        default:
            s = "Invalid NonlinearMethod: please submit bug report";
    }
    return s;
}

// Make a lowercase copy of s:
string lowercase(const string& s) {
    char* buf = new char[s.length()];
    s.copy(buf, s.length());
    for (uint i = 0; i < s.length(); i++)
        buf[i] = tolower(buf[i]);
    string r(buf, s.length());
    delete[] buf;
    return r;
}

VelocityScale s2velocityscale(const std::string& s_) {
    VelocityScale v;
    string s = lowercase(s_);
    if (s.find("wall") != string::npos)
        v = WallScale;
    else if (s.find("parab") != string::npos)
        v = ParabolicScale;
    // else if (s.find("bulk")  != string::npos) v = BulkScale;
    else {
        cerr << "warning : s2velocityscale(string) : unrecognized string " << s << endl;
        exit(1);
    }
    return v;
}

BaseFlow s2baseflow(const std::string& s_) {
    string s = lowercase(s_);
    BaseFlow b;
    if (s.find("zero") != string::npos)
        b = ZeroBase;
    else if (s.find("laminar") != string::npos)
        b = LaminarBase;
    else if (s.find("couette") != string::npos || s.find("linear") != string::npos)
        b = LinearBase;
    else if (s.find("parabolic") != string::npos || s.find("poiseuille") != string::npos)
        b = ParabolicBase;
    else if (s.find("ASBL") != string::npos || s.find("asbl") != string::npos || s.find("suction") != string::npos)
        b = SuctionBase;
    else if (s.find("arbitrary") != string::npos)
        b = ArbitraryBase;
    else {
        cerr << "warning : s2baseflow(string) : unrecognized string " << s << endl;
        exit(1);
    }
    return b;
}

MeanConstraint s2constraint(const std::string& s_) {
    MeanConstraint m;
    string s = lowercase(s_);
    if (s.find("pressure") != string::npos || s.find("gradp") != string::npos)
        m = PressureGradient;
    else if (s.find("velocity") != string::npos || s.find("bulkv") != string::npos)
        m = BulkVelocity;
    else {
        cerr << "warning : s2constraint(string) : unrecognized string " << s << endl;
        exit(1);
    }
    return m;
}

TimeStepMethod s2stepmethod(const string& s_) {
    TimeStepMethod step = SBDF3;
    string s = lowercase(s_);
    if (s.find("cnfe1") != string::npos)
        step = CNFE1;
    else if (s.find("cnab2") != string::npos)
        step = CNAB2;
    else if (s.find("cnrk2") != string::npos)
        step = CNRK2;
    else if (s.find("smrk2") != string::npos)
        step = SMRK2;
    else if (s.find("sbdf1") != string::npos)
        step = SBDF1;
    else if (s.find("sbdf2") != string::npos)
        step = SBDF2;
    else if (s.find("sbdf3") != string::npos)
        step = SBDF3;
    else if (s.find("sbdf4") != string::npos)
        step = SBDF4;
    else {
        cerr << "warning : s2stepstepod(string) : unrecognized string " << s << endl;
        exit(1);
    }
    return step;
}

NonlinearMethod s2nonlmethod(const string& s_) {
    NonlinearMethod nonl = Rotational;
    string s = lowercase(s_);
    if (s.find("rot") != string::npos)
        nonl = Rotational;
    else if (s.find("conv") != string::npos)
        nonl = Convection;
    else if (s.find("skew") != string::npos)
        nonl = SkewSymmetric;
    else if (s.find("alt") != string::npos)
        nonl = Alternating;
    else if (s.find("div") != string::npos)
        nonl = Divergence;
    else if (s.find("linear") != string::npos)
        nonl = LinearAboutProfile;
    else {
        cerr << "warning : s2nonlmethod(string) : unrecognized string " << s << endl;
        exit(1);
    }
    return nonl;
}

Dealiasing s2dealiasing(const std::string& s_) {
    Dealiasing d = DealiasXZ;
    string s = lowercase(s_);
    if (s.find("nodealiasing") != string::npos)
        d = NoDealiasing;
    else if (s.find("dealiasxz") != string::npos)
        d = DealiasXZ;
    else if (s.find("dealiasy") != string::npos)
        d = DealiasY;
    else if (s.find("dealiasxyz") != string::npos)
        d = DealiasXYZ;
    else {
        cerr << "warning : s2dealiasing(string) : unrecognized string " << s << endl;
        exit(1);
    }
    return d;
}

Verbosity s2verbosity(const std::string& s_) {
    Verbosity v = Silent;
    string s = lowercase(s_);
    if (s.find("silent") != string::npos)
        v = Silent;
    else if (s.find("time") != string::npos)
        v = PrintTime;
    else if (s.find("ticks") != string::npos)
        v = PrintTicks;
    else if (s.find("tausolve") != string::npos)
        v = VerifyTauSolve;
    else if (s == "all")
        v = VerifyTauSolve;
    else {
        cerr << "warning : s2verbosity(string) : unrecognized string " << s << endl;
        exit(1);
    }
    return v;
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

void DNSFlags::save(const string& outdir) const {
    if (mpirank() == 0) {
        string filename = appendSuffix(outdir, "dnsflags.txt");
        ofstream os(filename.c_str());
        if (!os.good())
            cferror("DNSFlags::save(outdir) :  can't open file " + filename);
        os.precision(16);
        os.setf(ios::left);
        os << setw(REAL_IOWIDTH) << nu << "  %nu\n"
           << setw(REAL_IOWIDTH) << Vsuck << "  %Vsuck\n"
           << setw(REAL_IOWIDTH) << rotation << "  %rotation\n"
           << setw(REAL_IOWIDTH) << theta << "  %theta\n"
           << setw(REAL_IOWIDTH) << dPdx << "  %dPdx\n"
           << setw(REAL_IOWIDTH) << dPdz << "  %dPdz\n"
           << setw(REAL_IOWIDTH) << Ubulk << "  %Ubulk\n"
           << setw(REAL_IOWIDTH) << Wbulk << "  %Wbulk\n"
           << setw(REAL_IOWIDTH) << Uwall << "  %Uwall\n"
           << setw(REAL_IOWIDTH) << uupperwall << "  %uupperwall\n"
           << setw(REAL_IOWIDTH) << ulowerwall << "  %ulowerwall\n"
           << setw(REAL_IOWIDTH) << wupperwall << "  %wupperwall\n"
           << setw(REAL_IOWIDTH) << wlowerwall << "  %wlowerwall\n"
           << setw(REAL_IOWIDTH) << t0 << "  %t0\n"
           << setw(REAL_IOWIDTH) << T << "  %T\n"
           << setw(REAL_IOWIDTH) << dT << "  %dT\n"
           << setw(REAL_IOWIDTH) << dt << "  %dt\n"
           << setw(REAL_IOWIDTH) << variabledt << "  %variabledt\n"
           << setw(REAL_IOWIDTH) << dtmin << "  %dtmin\n"
           << setw(REAL_IOWIDTH) << dtmax << "  %dtmax\n"
           << setw(REAL_IOWIDTH) << CFLmin << "  %CFLmin\n"
           << setw(REAL_IOWIDTH) << CFLmax << "  %CFLmax\n"
           << setw(REAL_IOWIDTH) << baseflow << "  %baseflow\n"
           << setw(REAL_IOWIDTH) << constraint << "  %constraint\n"
           << setw(REAL_IOWIDTH) << timestepping << "  %timestepping\n"
           << setw(REAL_IOWIDTH) << initstepping << "  %initstepping\n"
           << setw(REAL_IOWIDTH) << nonlinearity << "  %nonlinearity\n"
           << setw(REAL_IOWIDTH) << symmetryprojectioninterval << "  %symmetryprojectioninterval\n"
           << setw(REAL_IOWIDTH) << dealiasing << "  %dealiasing\n"
           << setw(REAL_IOWIDTH) << taucorrection << "  %taucorrection\n"
           << setw(REAL_IOWIDTH) << (bodyforce ? "nonzero_bodyforce" : "zero_bodyforce") << "  %bodyforce\n"
           << setw(REAL_IOWIDTH) << verbosity << "  %verbosity\n";
        os.unsetf(ios::left);
    }
}

void DNSFlags::load(int taskid, const string indir) {
    ifstream is;
    if (taskid == 0) {
        is.open(indir + "dnsflags.txt");
        if (!is.good()) {
            cout << "    DNSFlags::load(taskid, indir): can't open file " + indir + "dnsflags.txt" << endl;
            return;
        }
        if (!checkFlagContent(is, getFlagList())) {
            cerr << " DNSFlags::load(taskid, indir): the order of variables in the file is not what we expect !!"
                 << endl;
            exit(1);
        }
    }
    nu = getRealfromLine(taskid, is);
    Vsuck = getRealfromLine(taskid, is);
    rotation = getRealfromLine(taskid, is);
    theta = getRealfromLine(taskid, is);
    dPdx = getRealfromLine(taskid, is);
    dPdz = getRealfromLine(taskid, is);
    Ubulk = getRealfromLine(taskid, is);
    Wbulk = getRealfromLine(taskid, is);
    Uwall = getRealfromLine(taskid, is);
    uupperwall = getRealfromLine(taskid, is);
    ulowerwall = getRealfromLine(taskid, is);
    wupperwall = getRealfromLine(taskid, is);
    wlowerwall = getRealfromLine(taskid, is);
    t0 = getIntfromLine(taskid, is);
    T = getIntfromLine(taskid, is);
    dT = getIntfromLine(taskid, is);
    dt = getRealfromLine(taskid, is);
    variabledt = getIntfromLine(taskid, is);
    dtmin = getRealfromLine(taskid, is);
    dtmax = getRealfromLine(taskid, is);
    CFLmin = getRealfromLine(taskid, is);
    CFLmax = getRealfromLine(taskid, is);
    baseflow = parse_enum_value(is, s2baseflow, ZeroBase);
    constraint = parse_enum_value(is, s2constraint, PressureGradient);
    timestepping = parse_enum_value(is, s2stepmethod, SBDF3);
    initstepping = parse_enum_value(is, s2stepmethod, SBDF3);
    nonlinearity = parse_enum_value(is, s2nonlmethod, Rotational);
    symmetryprojectioninterval = getIntfromLine(taskid, is);
    dealiasing = parse_enum_value(is, s2dealiasing, DealiasXZ);
    taucorrection = getIntfromLine(taskid, is);
    getBodyforcefromLine(taskid, is);  // calling this function returns either error or zero. Thus, next line.
    bodyforce = (BodyForce*)0;
    verbosity = parse_enum_value(is, s2verbosity, Silent);
}

const vector<string> DNSFlags::getFlagList() {
    const vector<string> flagList = {"%nu",
                                     "%Vsuck",
                                     "%rotation",
                                     "%theta",
                                     "%dPdx",
                                     "%dPdz",
                                     "%Ubulk",
                                     "%Wbulk",
                                     "%Uwall",
                                     "%uupperwall",
                                     "%ulowerwall",
                                     "%wupperwall",
                                     "%wlowerwall",
                                     "%t0",
                                     "%T",
                                     "%dT",
                                     "%dt",
                                     "%variabledt",
                                     "%dtmin",
                                     "%dtmax",
                                     "%CFLmin",
                                     "%CFLmax",
                                     "%baseflow",
                                     "%constraint",
                                     "%timestepping",
                                     "%initstepping",
                                     "%nonlinearity",
                                     "%symmetryprojectioninterval",
                                     "%dealiasing",
                                     "%taucorrection",
                                     "%bodyforce",
                                     "%verbosity"};
    return flagList;
}

// TimeStep class
//====================================================================
ostream& operator<<(ostream& os, const TimeStep& dt) {
    os << "{dt=" << dt.dt() << ", n=" << dt.n() << ", dT=" << dt.dT() << ", N=" << dt.N() << ", dtmin=" << dt.dtmin()
       << ", dtmax=" << dt.dtmax() << ", CFLmin=" << dt.CFLmin() << ", CFL=" << dt.CFL() << ", CFLmax=" << dt.CFLmax()
       << ", variable=" << dt.variable() << "}";
    return os;
}

TimeStep::TimeStep()
    : n_(0),
      N_(0),
      dt_(0),
      dtmin_(0),
      dtmax_(0),
      dT_(0.0),
      T_(0.0),
      CFLmin_(0.0),
      CFL_(0.0),  // will take on meaninful value after first adjust
      CFLmax_(0.0),
      variable_(false) {}

TimeStep::TimeStep(Real dt, Real dtmin, Real dtmax, Real dT, Real CFLmin, Real CFLmax, bool variable)
    : n_(0),
      N_(0),
      dt_(dt),
      dtmin_(dtmin),
      dtmax_(dtmax),
      dT_(dT),
      T_(0.0),
      CFLmin_(CFLmin),
      CFL_((CFLmax + CFLmin) / 2),  // will take on meaningful value after first adjust
      CFLmax_(CFLmax),
      variable_(variable) {
    if (dtmin < 0 || dt < dtmin || dtmax < dt) {
        cerr << "error in TimeStep::TimeStep(dt, dtmin, dtmax, dT, CFLmin, CFLmax, variable) :\n"
             << "condition 0 <= dtmin <= dt <= dtmax does not hold" << endl;
        exit(1);
    }
    if (CFLmin < 0 || CFLmax < CFLmin) {
        cerr << "error in TimeStep::TimeStep(dt, dtmin, dtmax, dT, CFLmin, CFLmax, variable) :\n"
             << "condition 0 <= CFLmin <= CFLmax does not hold" << endl;
        exit(1);
    }
    if (dT < dtmin) {
        cerr << "error in TimeStep::TimeStep(dt, dtmin, dtmax, dT, CFLmin, CFLmax, variable) :\n"
             << "dT < dtmin" << endl;
        exit(1);
    }

    // Adjust dt to be integer divisor of dT. At this point we have 0 <= dtmin <= dt and dtmin <= dT
    n_ = Greater(iround(dT / dt), 1);
    dt_ = dT_ / n_;  // 0 <= dt <= dT and dtmin <= dT

    // Bump up or down to get within dtmin, dtmax
    while (dt_ < dtmin_ && n_ >= 2 && dT_ != 0) {
        dt_ = dT_ / --n_;  // guaranteed to terminate at  dtmin <= dt == dT
    }

    while (dt_ > dtmax_ && n_ <= INT_MAX && dT_ != 0) {
        dt_ = dT_ / ++n_;  // guaranteed to terminate at  dt == dT/INT_MAX
    }
    assert(dt_ > 0 && dt_ <= dT);
    assert(dt_ >= dtmin && dt <= dtmax);
}

TimeStep::TimeStep(DNSFlags& flags)
    // delegating constructor
    : TimeStep(flags.dt, flags.dtmin, flags.dtmax, flags.dT, flags.CFLmin, flags.CFLmax, flags.variabledt) {}

// relations
// n*dt = dT
bool TimeStep::adjust(Real CFL, bool verbose, ostream& os) {
    CFL_ = CFL;
    if (variable_ && (CFL <= CFLmin_ || CFL >= CFLmax_))
        return adjustToMiddle(CFL, verbose, os);
    else
        return false;
}

bool TimeStep::adjustToMiddle(Real CFL, bool verbose, ostream& os) {
    verbose = (verbose && (mpirank() == 0));

    if (dtmin_ == dtmax_ || dT_ == 0.0)
        return false;

    // New update algorithm puts CFL at midpoint btwn bounds
    // Aim for      CFL' == (CFLmax+CFLmin)/2
    // Change is    CFL' == CFL * dt'/dt
    // (CFLmax+CFLmin)/2 == CFL * n/n'      since dt=dT/n
    // So             n' == 2 n CFL/(CFLmax+CFLmin)
    //
    int n = Greater(iround(2 * n_ * CFL / (CFLmax_ + CFLmin_)), 1);
    Real dt = dT_ / n;

    // Bump dt up or down to get within [dtmin, dtmax]
    while (dt < dtmin_ && dt < dT_)
        dt = dT_ / --n;  // guaranteed to terminate at  dtmin <= dt == dT
    while (dt > dtmax_ && n <= INT_MAX)
        dt = dT_ / ++n;  // guaranteed to terminate at  dtmin <= dt == dT

    CFL *= dt / dt_;

    // Check to see if adjustment took dt out of range
    if (verbose && (CFL > CFLmax_ || CFL < CFLmin_)) {
        os << "TimeStep::adjust(CFL) : dt " << (CFL > CFLmax_ ? "bottomed" : "topped") << " out at\n"
           << " dt  == " << dt << endl
           << " CFL == " << CFL << endl
           << " n   == " << n << endl;
    }

    // If final choice for n differs from original n_, reset internal values
    bool adjustment = (n == n_) ? false : true;
    if (adjustment) {
        if (verbose) {
            os << "TimeStep::adjust(CFL) { " << endl;
            os << "   n : " << n_ << " -> " << n << endl;
            os << "  dt : " << dt_ << " -> " << dt << endl;
            os << " CFL : " << CFL_ << " -> " << CFL << endl;
            os << "}" << endl;
        }
        n_ = n;
        dt_ = dt;
        CFL_ = CFL;
    }
    return adjustment;
}

// to bring any variable*dt lower than a maximum
bool TimeStep::adjust(Real a, Real a_max, bool verbose, ostream& os) {
    if (variable_ && a >= a_max)
        return adjustToDesired(a, a_max, verbose, os);
    else
        return false;
}

bool TimeStep::adjustToDesired(Real a, Real a_des, bool verbose, ostream& os) {
    Real ai = a;

    verbose = (verbose && (mpirank() == 0));

    if (dtmin_ == dtmax_ || dT_ == 0.0)
        return false;

    // New update algorithm puts a at midpoint btwn bounds of a
    // Aim for      a' == (a_max+a_min)/2
    // Change is    a' == a * dt'/dt
    // (a_max+a_min)/2 == a * n/n'      since dt=dT/n
    // So             n' == 2 n a/(a_max+a_min)
    //
    int n = Greater(iround(n_ * a / a_des), 1);
    Real dt = dT_ / n;

    // Bump dt up or down to get within [dtmin, dtmax]
    while (dt < dtmin_ && dt < dT_)
        dt = dT_ / --n;  // guaranteed to terminate at  dtmin <= dt == dT
    while (dt > dtmax_ && n <= INT_MAX)
        dt = dT_ / ++n;  // guaranteed to terminate at  dtmin <= dt == dT

    a *= dt / dt_;

    // Check to see if adjustment took dt out of range
    if (verbose && (a > a_des)) {
        os << "TimeStep::adjust(a) : dt bottomed out at\n"
           << " dt  == " << dt << endl
           << " a   == " << a << endl
           << " n   == " << n << endl;
    }

    // If final choice for n differs from original n_, reset internal values
    bool adjustment = (n == n_) ? false : true;
    if (adjustment) {
        if (verbose) {
            os << "TimeStep::adjust(a) { " << endl;
            os << "   n : " << n_ << " -> " << n << endl;
            os << "  dt : " << dt_ << " -> " << dt << endl;
            os << "   a : " << ai << " -> " << a << endl;
            os << "}" << endl;
        }
        n_ = n;
        dt_ = dt;
    }
    return adjustment;
}

bool TimeStep::adjust_for_T(Real T, bool verbose, ostream& os) {
    verbose = (verbose && (mpirank() == 0));

    T_ = T;
    if (T < 0) {
        cerr << "TimeStep::adjust_for_T : can't integrate backwards in time.\n"
             << "Exiting." << endl;
        exit(1);
    }
    if (T == 0) {
        bool adjustment = (dt_ == 0) ? false : true;
        dt_ = 0;
        n_ = 0;
        dT_ = 0;
        T_ = 0;
        return adjustment;
    }
    int N = Greater(iround(T / dT_), 1);
    Real dT = T / N;
    int n = Greater(iround(dT / dt_), 1);
    Real dt = dT / n;

    while (dt < dtmin_ && n > 2 && dT != 0)
        dt = dT / --n;
    while (dt > dtmax_ && n <= INT_MAX && dT != 0)
        dt = dT / ++n;

    Real CFL = dt * CFL_ / dt_;

    bool adjustment = (dt == dt_) ? false : true;
    if (adjustment && verbose) {
        os << "TimeStep::adjust_for_T(Real T) { " << endl;
        os << "   T : " << T << endl;
        os << "  dT : " << dT_ << " -> " << dT << endl;
        os << "  dt : " << dt_ << " -> " << dt << endl;
        os << "  n  : " << n_ << " -> " << n << endl;
        os << "  N  : " << N_ << " -> " << N << endl;
        os << " CFL : " << CFL_ << " -> " << CFL << endl;
        os << "}" << endl;
    }
    n_ = n;
    N_ = N;
    dt_ = dt;
    dT_ = dT;
    CFL_ = CFL;

    return adjustment;
}

int TimeStep::n() const { return n_; }
int TimeStep::N() const { return N_; }
Real TimeStep::dt() const { return dt_; }
Real TimeStep::dT() const { return dT_; }
Real TimeStep::T() const { return T_; }
Real TimeStep::dtmin() const { return dtmin_; }
Real TimeStep::dtmax() const { return dtmax_; }
Real TimeStep::CFL() const { return CFL_; }
Real TimeStep::CFLmin() const { return CFLmin_; }
Real TimeStep::CFLmax() const { return CFLmax_; }
bool TimeStep::variable() const { return variable_; }
TimeStep::operator Real() const { return dT_ / n_; }

//====================================================================

ostream& operator<<(ostream& os, const DNSFlags& flags) {
    string s(", ");
    string tau = (flags.taucorrection) ? "TauCorrection" : "NoTauCorrection";
    const int p = os.precision();
    os.precision(16);
    os << "nu==" << flags.nu << s << "Vsuck==" << flags.Vsuck << s << "rotation==" << flags.rotation << s
       << "theta==" << flags.theta << s << "dPdx==" << flags.dPdx << s << "dPdz==" << flags.dPdz << s
       << "Ubulk==" << flags.Ubulk << s << "Wbulk==" << flags.Wbulk << s << "uwall==" << flags.Uwall << s
       << "uupper==" << flags.uupperwall << s << "ulower==" << flags.ulowerwall << s << "wupper==" << flags.wupperwall
       << s << "wlower==" << flags.wlowerwall << s << "t0==" << flags.t0 << s << "dT==" << flags.dT << s
       << "dt==" << flags.dt << s << "variabledt==" << flags.variabledt << s << "dtmin==" << flags.dtmin << s
       << "dtmax==" << flags.dtmax << s << "CFLmin==" << flags.CFLmin << s << "CFLmax==" << flags.CFLmax << s
       << flags.baseflow << s << flags.constraint << s << flags.timestepping << s << flags.initstepping << s
       << flags.nonlinearity << s << flags.dealiasing << s << (flags.bodyforce ? "nonzero_bodyforce" : "zero_bodyforce")
       << s << tau << s << flags.verbosity;
    os.precision(p);
    return os;
}

}  // namespace chflow
