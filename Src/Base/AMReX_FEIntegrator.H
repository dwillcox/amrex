#ifndef AMREX_FE_INTEGRATOR_H
#define AMREX_FE_INTEGRATOR_H
#include <functional>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#include "AMReX_IntegratorBase.H"

template<class T>
class FEIntegrator : public IntegratorBase<T>
{
private:
    amrex::Vector<T> F_nodes;

public:
    FEIntegrator(T& S_old, T& S_new, amrex::Real time) :
                    IntegratorBase(S_old, S_new, time)
    {
        IntegratorOps::EmplaceLike(F_nodes, S_old);
    }

    amrex::Real advance(const amrex::Real timestep)
    {
        // Assume before advance() that S_new is valid data at the current time ("time" argument)
        // So we update S_old by copying the new state.
        IntegratorOps::Copy(S_old, S_new);

        // F = RHS(S, t)
        T& F = F_nodes[0];
        rhs(F, S_new, time);

        // S_new += timestep * dS/dt
        IntegratorOps::Saxpy(S_new, timestep, F);

        // Call the post-update hook for S_new
        post_update(S_new);

        // Update time
        time += timestep;

        // Return timestep
        return timestep;
    }
};

#endif
