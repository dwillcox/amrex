#ifndef AMREX_RK_INTEGRATOR_H
#define AMREX_RK_INTEGRATOR_H
#include <functional>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include "AMReX_IntegratorBase.H"

namespace ButcherTableauTypes {
    enum ButcherTypes {User = 0, ForwardEuler, Trapezoid, SSPRK3, RK4, NumTypes};
};

template<class T>
class RKIntegrator : public IntegratorBase<T>
{
private:
    int tableau_type;
    int number_nodes;
    bool use_adaptive_timestep;
    amrex::Vector<T> F_nodes;
    amrex::Vector<amrex::Vector<amrex::Real> > tableau;
    amrex::Vector<amrex::Real> weights;
    amrex::Vector<amrex::Real> extended_weights;
    amrex::Vector<amrex::Real> nodes;
    void initialize_preset_tableau();
    void initialize_parameters();
    void initialize_stages();

public:
    RKIntegrator(T& S_old_external, T& S_new_external, amrex::Real time);
    amrex::Real advance(const amrex::Real timestep);
};

#endif
