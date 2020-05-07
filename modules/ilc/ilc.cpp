/**
 * This file is a part of channelflow version 2.0, https://channelflow.ch .
 * License is GNU GPL version 2 or later: ./LICENSE
 *
 * Original author: Florian Reetz
 */

#include "modules/ilc/ilc.h"


namespace chflow {

int field2vector_size(const FlowField& u, const FlowField& temp) {
    int Kx = u.kxmaxDealiased();
    int Kz = u.kzmaxDealiased();
    int Ny = u.Ny();
    // FIXME: Determine array size
    // The original formula was
    //     int N = 4* ( Kx+2*Kx*Kz+Kz ) * ( Ny-3 ) +2* ( Ny-2 );
    // but I've not been able to twist my head enough to adapt it do distributed FlowFields.
    // Since it doesn't take that much time, we now perform the loops twice, once empty to
    // determine cfarray sizes and once with the actual data copying. // Tobias
    int N = 0;
    if (u.taskid() == u.task_coeff(0, 0))
        N += 2 * (Ny - 2);
    for (int kx = 1; kx <= Kx; ++kx)
        if (u.taskid() == u.task_coeff(u.mx(kx), 0))
            N += 2 * (Ny - 2) + 2 * (Ny - 4);
    for (int kz = 1; kz <= Kz; ++kz)
        if (u.taskid() == u.task_coeff(0, u.mz(kz)))
            N += 2 * (Ny - 2) + 2 * (Ny - 4);
    for (int kx = -Kx; kx <= Kx; ++kx) {
        if (kx == 0)
            continue;
        int mx = u.mx(kx);
        for (int kz = 1; kz <= Kz; ++kz) {
            int mz = u.mz(kz);
            if (u.taskid() == u.task_coeff(mx, mz)) {
                N += 2 * (Ny - 2) + 2 * (Ny - 4);
            }
        }
    }

    if (temp.taskid() == temp.task_coeff(0, 0))
        N += Ny;
    for (int kx = 1; kx <= Kx; ++kx)
        if (temp.taskid() == temp.task_coeff(temp.mx(kx), 0))
            N += 2 * Ny;
    for (int kz = 1; kz <= Kz; ++kz)
        if (temp.taskid() == temp.task_coeff(0, temp.mz(kz)))
            N += 2 * Ny;
    for (int kx = -Kx; kx <= Kx; kx++) {
        if (kx == 0)
            continue;
        for (int kz = 1; kz <= Kz; kz++) {
            if (temp.taskid() == temp.task_coeff(temp.mx(kx), temp.mz(kz))) {
                N += 2 * Ny;
            }
        }
    }

    return N;
}

/** \brief Turn the two flowfields for velocity and temperature into one Eigen vector
 * \param[in] u velocity field
 * \param[in] temp temperature field
 * \param[in] a vector for the linear algebra
 *
 * The vectorization of u is analog to the field2vector, temparture is piped entirely
 * into the vector (a single independent dimensions)
 */
void field2vector(const FlowField& u, const FlowField& temp, Eigen::VectorXd& a) {
    Eigen::VectorXd b;
    assert(temp.xzstate() == Spectral && temp.ystate() == Spectral);
    field2vector(u, b);
    int Kx = temp.kxmaxDealiased();
    int Kz = temp.kzmaxDealiased();
    int Ny = temp.Ny();

    int n = field2vector_size(u, temp);  // b.size() +6*Kx*Kz*Ny;

    if (a.size() < n)
        a.resize(n, true);
    setToZero(a);
    int pos = b.size();
    a.topRows(pos) = b;

    // (0,:,0)
    for (int ny = 0; ny < Ny; ++ny)
        if (temp.taskid() == temp.task_coeff(0, 0))
            a(pos++) = Re(temp.cmplx(0, ny, 0, 0));

    for (int kx = 1; kx <= Kx; ++kx) {
        int mx = temp.mx(kx);
        if (temp.taskid() == temp.task_coeff(mx, 0)) {
            for (int ny = 0; ny < Ny; ++ny) {
                a(pos++) = Re(temp.cmplx(mx, ny, 0, 0));
                a(pos++) = Im(temp.cmplx(mx, ny, 0, 0));
            }
        }
    }
    for (int kz = 1; kz <= Kz; ++kz) {
        int mz = temp.mz(kz);
        if (temp.taskid() == temp.task_coeff(0, mz)) {
            for (int ny = 0; ny < Ny; ++ny) {
                a(pos++) = Re(temp.cmplx(0, ny, mz, 0));
                a(pos++) = Im(temp.cmplx(0, ny, mz, 0));
            }
        }
    }
    for (int kx = -Kx; kx <= Kx; kx++) {
        if (kx == 0)
            continue;
        int mx = temp.mx(kx);
        for (int kz = 1; kz <= Kz; kz++) {
            int mz = temp.mz(kz);
            if (u.taskid() == u.task_coeff(mx, mz)) {
                for (int ny = 0; ny < Ny; ny++) {
                    a(pos++) = Re(temp.cmplx(mx, ny, mz, 0));
                    a(pos++) = Im(temp.cmplx(mx, ny, mz, 0));
                }
            }
        }
    }
}

/** \brief Turn  one Eigen vector into the two flowfields for velocity and temperature
 * \param[in] a vector for the linear algebra
 * \param[in] u velocity field
 * \param[in] temp temperature field
 *
 * The vectorization of u is analog to the field2vector, temperature is piped entirely
 * into the vector (a single independent dimension)
 */
void vector2field(const Eigen::VectorXd& a, FlowField& u, FlowField& temp) {
    assert(temp.xzstate() == Spectral && temp.ystate() == Spectral);
    temp.setToZero();
    Eigen::VectorXd b;
    int N = field2vector_size(u);
    int Kx = temp.kxmaxDealiased();
    int Kz = temp.kzmaxDealiased();
    int Ny = temp.Ny();
    b = a.topRows(N);
    vector2field(b, u);

    int pos = N;
    double reval, imval;
    Complex val;

    if (temp.taskid() == temp.task_coeff(0, 0))
        for (int ny = 0; ny < Ny; ++ny)
            temp.cmplx(0, ny, 0, 0) = Complex(a(pos++), 0);

    for (int kx = 1; kx <= Kx; ++kx) {
        int mx = temp.mx(kx);
        if (temp.taskid() == temp.task_coeff(mx, 0)) {
            for (int ny = 0; ny < Ny; ++ny) {
                reval = a(pos++);
                imval = a(pos++);
                temp.cmplx(mx, ny, 0, 0) = Complex(reval, imval);
            }
        }

        // ------------------------------------------------------
        // Now copy conjugates of u(kx,ny,0,i) to u(-kx,ny,0,i). These are
        // redundant modes stored only for the convenience of FFTW.
        int mxm = temp.mx(-kx);
        int send_id = temp.task_coeff(mx, 0);
        int rec_id = temp.task_coeff(mxm, 0);
        for (int ny = 0; ny < Ny; ++ny) {
            if (temp.taskid() == send_id && send_id == rec_id) {  // all is on the same process -> just copy
                temp.cmplx(mxm, ny, 0, 0) = conj(temp.cmplx(mx, ny, 0, 0));
            }
#ifdef HAVE_MPI     // send_id != rec_id requires multiple processes
            else {  // Transfer the conjugates via MPI
                if (temp.taskid() == send_id) {
                    Complex tmp0 = conj(temp.cmplx(mx, ny, 0, 0));
                    MPI_Send(&tmp0, 1, MPI_DOUBLE_COMPLEX, rec_id, 0, MPI_COMM_WORLD);
                }
                if (u.taskid() == rec_id) {
                    Complex tmp0;
                    MPI_Status status;
                    MPI_Recv(&tmp0, 1, MPI_DOUBLE_COMPLEX, send_id, 0, MPI_COMM_WORLD, &status);
                    temp.cmplx(mxm, ny, 0, 0) = tmp0;
                }
            }
#endif
        }
    }
    for (int kz = 1; kz <= Kz; ++kz) {
        int mz = temp.mz(kz);
        if (temp.taskid() == temp.task_coeff(0, mz)) {
            for (int ny = 0; ny < Ny; ++ny) {
                reval = a(pos++);
                imval = a(pos++);
                temp.cmplx(0, ny, mz, 0) = Complex(reval, imval);
            }
        }
    }
    for (int kx = -Kx; kx <= Kx; kx++) {
        if (kx == 0)
            continue;
        int mx = temp.mx(kx);
        for (int kz = 1; kz <= Kz; kz++) {
            int mz = temp.mz(kz);
            if (u.taskid() == u.task_coeff(mx, mz)) {
                for (int ny = 0; ny < Ny; ny++) {
                    reval = a(pos++);
                    imval = a(pos++);
                    val = Complex(reval, imval);
                    temp.cmplx(mx, ny, mz, 0) = val;
                }
            }
        }
    }
    temp.setPadded(true);
}

// ILC::ILC()
//   :
//   DNS(){
//
// }

ILC::ILC(const std::vector<FlowField>& fields, const ILCFlags& flags)
    :  // base class constructor with no arguments is called automatically (see DNS::DNS())
      main_obe_(0),
      init_obe_(0) {
    main_obe_ = newOBE(fields, flags);
    // creates DNSAlgo with ptr of "nse"-daughter type "obe"
    main_algorithm_ = newAlgorithm(fields, main_obe_, flags);
    if (!main_algorithm_->full() && flags.initstepping != flags.timestepping) {
        ILCFlags initflags = flags;
        initflags.timestepping = flags.initstepping;
        initflags.dt = flags.dt;
        init_obe_ = newOBE(fields, flags);

        // creates DNSAlgo with ptr of "nse"-daughter type "obe"
        init_algorithm_ = newAlgorithm(fields, init_obe_, initflags);
        // Safety check
        if (init_algorithm_->Ninitsteps() != 0)
            std::cerr << "DNS::DNS(fields, flags) :\n"
                 << flags.initstepping << " can't initialize " << flags.timestepping
                 << " since it needs initialization itself.\n";
    }
}

ILC::~ILC() {}

const ChebyCoeff& ILC::Tbase() const {
    if (main_obe_)
        return main_obe_->Tbase();
    else if (init_obe_)
        return init_obe_->Tbase();
    else {
        std::cerr << "Error in ILC::Tbase():Tbase is currently undefined" << std::endl;
        exit(1);
        return init_obe_->Tbase();  // to make compiler happy
    }
}

const ChebyCoeff& ILC::Ubase() const {
    if (main_obe_)
        return main_obe_->Ubase();
    else if (init_obe_)
        return init_obe_->Ubase();
    else {
        std::cerr << "Error in ILC::Ubase(): Ubase is currently undefined" << std::endl;
        exit(1);
        return init_obe_->Ubase();  // to make compiler happy
    }
}

const ChebyCoeff& ILC::Wbase() const {
    if (main_obe_)
        return main_obe_->Wbase();
    else if (init_obe_)
        return init_obe_->Wbase();
    else {
        std::cerr << "Error in ILC::Wbase(): Wbase is currently undefined" << std::endl;
        exit(1);
        return init_obe_->Wbase();  // to make compiler happy
    }
}

std::shared_ptr<OBE> ILC::newOBE(const std::vector<FlowField>& fields, const ILCFlags& flags) {
    std::shared_ptr<OBE> obe(new OBE(fields, flags));
    return obe;
}

std::shared_ptr<OBE> ILC::newOBE(const std::vector<FlowField>& fields, const std::vector<ChebyCoeff>& base, const ILCFlags& flags) {
    std::shared_ptr<OBE> obe(new OBE(fields, base, flags));
    return obe;
}

}  // namespace channelflow