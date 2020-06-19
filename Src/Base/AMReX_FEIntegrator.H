#ifndef AMREX_FE_INTEGRATOR_H
#define AMREX_FE_INTEGRATOR_H
#include <functional>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include "AMReX_IntegratorBase.H"

template<class T>
class FEIntegrator : public IntegratorBase<T>
{
private:
    typedef IntegratorBase<T> BaseT;

    amrex::Vector<T> F_nodes;

public:
    FEIntegrator(T& S_old, T& S_new, amrex::Real time) :
                    IntegratorBase<T>(S_old, S_new, time)
    {
        IntegratorOps<T>::EmplaceLike(F_nodes, S_old);
    }

    amrex::Real advance(const amrex::Real timestep)
    {
        // Assume before advance() that S_old is valid data at the current time ("time" argument)
        // So we initialize S_new by copying the old state.
        T& S_old = BaseT::get_old_data();
        T& S_new = BaseT::get_new_data();

        IntegratorOps<T>::Copy(S_new, S_old);

        // F = RHS(S, t)
        amrex::Real& time = BaseT::get_time();
        T& F = F_nodes[0];
        BaseT::rhs(F, S_new, time);

        // S_new += timestep * dS/dt
        IntegratorOps<T>::Saxpy(S_new, timestep, F);

        // Update time
        time += timestep;

        // Call the post-update hook for S_new
        BaseT::post_update(S_new, time);

        // Return timestep
        return timestep;
    }
};

#endif
