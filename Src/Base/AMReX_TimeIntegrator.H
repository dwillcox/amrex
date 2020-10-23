#ifndef AMREX_TIME_INTEGRATOR_H
#define AMREX_TIME_INTEGRATOR_H
#include <functional>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_ParmParse.H>
#include "AMReX_IntegratorBase.H"
#include "AMReX_FEIntegrator.H"
#include "AMReX_RKIntegrator.H"

namespace IntegratorTypes {
    enum TimeIntegratorTypes {ForwardEuler = 0,
                              ExplicitRungeKutta};
};

template<class T>
class TimeIntegrator
{
private:
    amrex::Real time;
    amrex::Real timestep;
    int step_number;
    std::unique_ptr<IntegratorBase<T> > integrator_ptr;
    std::function<void ()> post_timestep;

public:
    TimeIntegrator(T& S_old_external, T& S_new_external,
                   amrex::Real initial_time = 0.0, int initial_step = 0)
    {
        amrex::ParmParse pp("integration");

        int integrator_type = IntegratorTypes::ForwardEuler;
        pp.query("type", integrator_type);

        switch (integrator_type)
        {
            case IntegratorTypes::ForwardEuler:
                integrator_ptr = std::make_unique<FEIntegrator<T> >(S_old_external, S_new_external, initial_time);
                break;
            case IntegratorTypes::ExplicitRungeKutta:
                integrator_ptr = std::make_unique<RKIntegrator<T> >(S_old_external, S_new_external, initial_time);
                break;
            default:
                amrex::Error("integration.type did not name a valid integrator type.");
                break;
        }

        // By default, do nothing post-timestep
        set_post_timestep([](){});

        // By default, do nothing after updating the state
        // In general, this is where BCs should be filled
        set_post_update([](T& S_data, amrex::Real S_time){});

        // By default, do nothing
        set_rhs([](T& S_rhs, const T& S_data, const amrex::Real time){});

        // Set the initial time and step number
        timestep = 0.0;
        time = initial_time;
        step_number = initial_step;
    }

    virtual ~TimeIntegrator() {}

    void set_post_timestep(std::function<void ()> F)
    {
        post_timestep = F;
    }

    void set_post_update(std::function<void (T&, amrex::Real)> F)
    {
        integrator_ptr->set_post_update(F);
    }

    void set_rhs(std::function<void(T&, const T&, const amrex::Real)> F)
    {
        integrator_ptr->set_rhs(F);
    }

    void advance(const amrex::Real dt)
    {
        integrator_ptr->advance(dt);
    }

    void integrate(const amrex::Real start_timestep, const amrex::Real end_time, const int nsteps)
    {
        amrex::Real timestep = start_timestep;
        bool stop_advance = false;
        for (; step_number < nsteps && !stop_advance; ++step_number)
        {
            if (end_time - time < timestep) {
                timestep = end_time - time;
                stop_advance = true;
            }

            if (step_number > 0) {
                std::swap(get_old_data(), get_new_data());
            }

            // Call the time integrator advance
            integrator_ptr->advance(timestep);

            // Update our time variable
            time = integrator_ptr->get_time();

            // Call the post-timestep hook
            post_timestep();
        }
    }

    void map_data(std::function<void(T&)> Map)
    {
        integrator_ptr->map_data(Map);
    }

    T& get_new_data()
    {
        return integrator_ptr->get_new_data();
    }

    T& get_old_data()
    {
        return integrator_ptr->get_old_data();
    }

    amrex::Real get_time()
    {
        return time;
    }

    amrex::Real get_timestep()
    {
        return timestep;
    }

    void set_timestep(amrex::Real dt)
    {
        timestep = dt;
    }

    int get_step_number()
    {
        return step_number;
    }

};

#endif
