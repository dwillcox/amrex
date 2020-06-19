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

    static void EmplaceLike(amrex::Vector<T>& V, const T& Other)
    {
        // Emplace a new T in V with the same size as Other and get a reference
        int i = V.size();
        V.emplace_back(Other.Geom(0), Other.ParticleDistributionMap(0), Other.ParticleBoxArray(0));
        T& pc = V[i];

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

    static void Saxpy(T& Y, const amrex::Real a, const T& X)
    {
        // Calculate Y += a * X for all the Real struct data in the particles.
        typedef amrex::ParIter<T::NStructReal, T::NStructInt, T::NArrayReal, T::NArrayInt> TParIter;
        typedef amrex::Particle<T::NStructReal, T::NStructInt> ParticleType;

        int lev = 0;
        TParIter pty(Y, lev);
        TParIter ptx(X, lev);

        auto checkValid = [&]() -> bool {
            bool pty_v = pty.isValid();
            bool ptx_v = ptx.isValid();
            AMREX_ASSERT(pty_v == ptx_v);
            return pty_v && ptx_v;
        };

        auto ptIncrement = [&](){ ++pty; ++ptx; };

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (; checkValid(); ptIncrement())
        {
            const int npy  = pty.numParticles();
            const int npx  = ptx.numParticles();
            AMREX_ASSERT(npy == npx);

            ParticleType* psy = &(pty.GetArrayOfStructs()[0]);
            ParticleType* psx = &(ptx.GetArrayOfStructs()[0]);

            ParallelFor ( npy, [=] AMREX_GPU_DEVICE (int i) {
                ParticleType& py = psy[i];
                ParticleType& px = psx[i];

                for (int j = 0; j < T::NStructReal; ++j) {
                    py.rdata(j) += a * px.rdata(j);
                }
            });
        }
    }

};

template<class T>
struct IntegratorOps<T, typename std::enable_if<std::is_same<amrex::MultiFab, T>::value>::type>
{

    static void EmplaceLike(amrex::Vector<T>& V, const T& Other, bool Grow = false)
    {
        // Emplace a new T in V with the same size as Other
        int nGrow = Grow ? Other.nGrow() : 0;
        V.emplace_back(Other.boxArray(), Other.DistributionMap(), Other.nComp(), nGrow);
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

    amrex::Real& get_time()
    {
        return time;
    }
};

#endif
