#ifndef AMREX_INTEGRATOR_BASE_H
#define AMREX_INTEGRATOR_BASE_H
#include <functional>
#include <type_traits>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Particles.H>

template<class T, typename Tv = void> struct IntegratorOps;

template<class T>
struct IntegratorOps<T, typename std::enable_if<std::is_base_of<amrex::ParticleContainerBase, T>::value>::type>
{

    static void CreateLike(amrex::Vector<std::unique_ptr<T> >& V, const T& Other)
    {
        // Emplace a new T in V with the same size as Other and get a reference
        V.emplace_back(std::make_unique<T>(Other.Geom(0), Other.ParticleDistributionMap(0), Other.ParticleBoxArray(0)));
        T& pc = *V[V.size()-1];

        // We want the particles to have all the same position, cpu, etc.
        // as in Other, so do a copy from Other to our new particle container.
        Copy(pc, Other);
    }

    static void Copy(T& Y, const T& Other)
    {
        // Copy the contents of Other into Y
        const bool local = true;
        Y.copyParticles(Other, local);
    }

    static void Saxpy(T& Y, const amrex::Real a, T& X)
    {
        // Calculate Y += a * X using a particle-level saxpy function supplied by the particle container T
        typedef typename T::ParIterType ParIterType;
        typedef typename T::ParticleType ParticleType;

        int lev = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (ParIterType pti(Y, lev); pti.isValid(); ++pti)
    {
        auto grid_tile = pti.GetPairIndex();

        auto& ptileY = Y.ParticlesAt(lev, grid_tile.first, grid_tile.second);
        auto& ptileX = X.ParticlesAt(lev, grid_tile.first, grid_tile.second);

        AMREX_ASSERT(ptileY.numParticles() == ptileX.numParticles());

        int np = ptileY.numParticles();

        ParticleType* psY = &(ptileY.GetArrayOfStructs()[0]);
        ParticleType* psX = &(ptileX.GetArrayOfStructs()[0]);

        auto particle_apply_rhs = T::particle_apply_rhs;

        amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) {
            ParticleType& py = psY[i];
            const ParticleType& px = psX[i];

            AMREX_ASSERT(py.id() == px.id());

            particle_apply_rhs(py, a, px);
        });
    }

    }

};

template<class T>
struct IntegratorOps<T, typename std::enable_if<std::is_same<amrex::MultiFab, T>::value>::type>
{

    static void CreateLike(amrex::Vector<std::unique_ptr<T> >& V, const T& Other, bool Grow = false)
    {
        // Emplace a new T in V with the same size as Other
        int nGrow = Grow ? Other.nGrow() : 0;
        V.emplace_back(std::make_unique<T>(Other.boxArray(), Other.DistributionMap(), Other.nComp(), nGrow));
    }

    static void Copy(T& Y, const T& Other, bool Grow = true)
    {
        // Copy the contents of Other into Y
        int nGrow = Grow ? Other.nGrow() : 0;
        amrex::MultiFab::Copy(Y, Other, 0, 0, Other.nComp(), nGrow);
    }

    static void Saxpy(T& Y, const amrex::Real a, const T& X, bool Grow = false)
    {
        // Calculate Y += a * X
        int nGrow = Grow ? X.nGrow() : 0;
        amrex::MultiFab::Saxpy(Y, a, X, 0, 0, X.nComp(), nGrow);
    }

};

template<class T>
class IntegratorBase
{
private:
    std::function<void(T&, const T&, const amrex::Real)> Fun;

protected:
    std::function<void (T&, amrex::Real)> post_update;
    T& S_old;
    T& S_new;
    amrex::Real time;

public:
    IntegratorBase(T& S_old_external, T& S_new_external, amrex::Real initial_time) :
                        time(initial_time),
                        S_old(S_old_external),
                        S_new(S_new_external) {}

    virtual ~IntegratorBase() {}

    void set_rhs(std::function<void(T&, const T&, const amrex::Real)> F)
    {
        Fun = F;
    }

    void set_post_update(std::function<void (T&, amrex::Real)> F)
    {
        post_update = F;
    }

    virtual amrex::Real advance(const amrex::Real timestep)
    {
        return timestep;
    }

    void rhs(T& S_rhs, const T& S_data, const amrex::Real time)
    {
        Fun(S_rhs, S_data, time);
    }

    T& get_new_data()
    {
        return S_new;
    }

    T& get_old_data()
    {
        return S_old;
    }

    virtual void map_data(std::function<void(T&)> Map)
    {
        Map(get_old_data());
        Map(get_new_data());
    }

    amrex::Real& get_time()
    {
        return time;
    }
};

#endif
