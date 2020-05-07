/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Original author: Florian Reetz
 */

#include "modules/ilc/ilcflags.h"

namespace chflow {

ILCFlags::ILCFlags(Real nu_, Real kappa_, Real alpha_, Real grav_, Real rho_ref_, Real t_ref_, Real gammax_,
                   Real gammaz_, Real ulowerwall_, Real uupperwall_, Real wlowerwall_, Real wupperwall_,
                   Real tlowerwall_, Real tupperwall_, Real ystats_)
    : kappa(kappa_),
      alpha(alpha_),
      grav(grav_),
      rho_ref(rho_ref_),
      t_ref(t_ref_),
      gammax(gammax_),
      gammaz(gammaz_),
      tlowerwall(tlowerwall_),
      tupperwall(tupperwall_),
      ystats(ystats_) {
    nu = nu_;
    ulowerwall = ulowerwall_;
    uupperwall = uupperwall_;
    wlowerwall = wlowerwall_;
    wupperwall = wupperwall_;

    //   freefall   = sqrt(grav*alpha*(tlowerwall-tupperwall));
    //   Treference = 0.5*(tlowerwall+tupperwall);
}

ILCFlags::ILCFlags(ArgList& args, const bool laurette) {
    // ILC system parameters
    args.section("System parameters");
    const Real Rayleigh_ = args.getreal("-Ra", "--Rayleigh", 4000, "pseudo-Rayleigh number == 1/(nu kp)");
    const Real Prandtl_ = args.getreal("-Pr", "--Prandtl", 1, "Prandtl number == nu/kp");
    const Real gammax_ = args.getreal("-gx", "--gammax", 45, "inclination angle of x-dim in deg");
    const Real gammaz_ = args.getreal("-gz", "--gammaz", 0, "inclination angle of z-dim in deg");
    const Real nuarg_ =
        args.getreal("-nu", "--nu", 1, "kinematic viscosity (takes precedence, with kp, over Ra and Pr, if nonunity)");
    const Real kparg_ = args.getreal("-kp", "--kappa", 1,
                                     "thermal conductivity (takes precedence, with nu, over Ra and Pr, if nonunity)");

    // define Channelflow boundary conditions from arglist
    args2BC(args);

    // add ILC boundary conditions
    const Real deltaT_ =
        args.getreal("-delT", "--DeltaT", 1, "temperature difference between the walls: DeltaT = T_lower - T_upper");
    const Real Tref_ = args.getreal("-Tref", "--Treference", 0,
                                    "reference temperature of the surrounding fluid reservoir for thermal expansion");
    const Real grav_ = args.getreal(
        "-grav", "--gravity", 1.0,
        "gravity coupling of the velocity fluctuations to the temperature fluctuations (no change of base or Ra)");
    const Real ilcUwall_ =
        args.getreal("-ilcUw", "--ilcUwall", 0.0, "in ILC, magnitude of imposed wall velocity, +/-Uwall at y = +/-h");

    // define Channelflow numerics from arglist
    args2numerics(args, laurette);

    // also needed for ILC but better show at the end
    const std::string tsymmstr = args.getstr("-tsymms", "--tempsymmetries", "",
                                        "constrain temp(t) to invariant "
                                        "symmetric subspace, argument is the filename for a file "
                                        "listing the generators of the isotropy group");
    const Real ystats_ = args.getreal("-ys", "--ystats", 0, "y-coordinate of height dependent statistics, e.g. Nu(y)");

    // set flags
    nu = (nuarg_ != 1) ? nuarg_ : sqrt(Prandtl_ / Rayleigh_);
    kappa = (kparg_ != 1) ? kparg_ : 1.0 / sqrt(Prandtl_ * Rayleigh_);
    gammax = gammax_ / 180.0 * pi;
    gammaz = gammaz_ / 180.0 * pi;
    t_ref = Tref_;
    tlowerwall = 0.5 * deltaT_;
    tupperwall = -0.5 * deltaT_;
    grav = grav_;
    ystats = ystats_;

    // overwrite the wall velocity with ilcUwall flag
    Uwall = ilcUwall_;
    ulowerwall = -ilcUwall_ * cos(theta);
    uupperwall = ilcUwall_ * cos(theta);
    wlowerwall = -ilcUwall_ * sin(theta);
    wupperwall = ilcUwall_ * sin(theta);

    if (tsymmstr.length() > 0) {
        SymmetryList tsymms(tsymmstr);
        tempsymmetries = tsymms;
    }

    save();
}

void ILCFlags::save(const std::string& savedir) const {
    DNSFlags::save(savedir);
    if (mpirank() == 0) {
        std::string filename = appendSuffix(savedir, "ilcflags.txt");
        std::ofstream os(filename.c_str());
        if (!os.good())
            cferror("ILCFlags::save(savedir) :  can't open file " + filename);
        os.precision(16);
        os.setf(std::ios::left);
        os << std::setw(REAL_IOWIDTH) << kappa << "  %kappa\n"
           << std::setw(REAL_IOWIDTH) << gammax << "  %gammax\n"
           << std::setw(REAL_IOWIDTH) << gammaz << "  %gammaz\n"
           << std::setw(REAL_IOWIDTH) << alpha << "  %alpha\n"
           << std::setw(REAL_IOWIDTH) << grav << "  %grav\n"
           << std::setw(REAL_IOWIDTH) << rho_ref << "  %rho_ref\n"
           << std::setw(REAL_IOWIDTH) << t_ref << "  %t_ref\n"
           << std::setw(REAL_IOWIDTH) << tupperwall << "  %tupperwall\n"
           << std::setw(REAL_IOWIDTH) << tlowerwall << "  %tlowerwall\n"
           << std::setw(REAL_IOWIDTH) << ystats << "  %ystats\n";
        os.unsetf(std::ios::left);
    }
}

void ILCFlags::load(int taskid, const std::string indir) {
    DNSFlags::load(taskid, indir);
    std::ifstream is;
    if (taskid == 0) {
        is.open(indir + "ilcflags.txt");
        if (!is.good())
            cferror(" ILCFlags::load(taskid, flags, dt, indir):  can't open file " + indir + "ilcflags.txt");
    }
    kappa = getRealfromLine(taskid, is);
    gammax = getRealfromLine(taskid, is);
    gammaz = getRealfromLine(taskid, is);
    alpha = getRealfromLine(taskid, is);
    grav = getRealfromLine(taskid, is);
    rho_ref = getRealfromLine(taskid, is);
    t_ref = getRealfromLine(taskid, is);
    tupperwall = getRealfromLine(taskid, is);
    tlowerwall = getRealfromLine(taskid, is);
    ystats = getRealfromLine(taskid, is);
}

}  // namespace channelflow
