#include <cstdio>
#include <ctime>
#include <iostream>
#include <sstream>
#include <limits>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include <iomanip>

#include "Type.h"
#include "Config.h"
#include "SIMD.h"
#include "Workspace.h"

// algorithms
#include "Test-Classic.h"
#include "Test-ClassicMod.h"
#include "Test-ClassicOffset.h"
#include "Test-BitSet.h"
#include "Test-BitSetNoPad.h"
#include "Test-MKL.h"
#include "Test-Direct.h"
#include "Test-Direct2.h"
#include "Test-DirectCache.h"
#include "Test-Eytzinger.h"

#include "Unroller.h"

using namespace std;

template <typename T, InstrSet I>
class Tester
{
public:
    template <bool Check, typename ALGO>
    NO_INLINE
    static void singlerun(DataWorkspace<T>& p, const ALGO& info)
    {
        uint32 n = static_cast<uint32>(p.m_z.size());
        Loop<I,ALGO>::loop(info, p.m_r.begin(), p.m_z.begin(), n);
    }
};


template <typename T>
class Tester<T, Scalar>
{    
    template <typename ALGO>
    NO_INLINE
    static uint32 run(const ALGO& info, T zi)
    {
        return info.scalar(zi);
    }

public:
    template <bool Check, typename ALGO>
    FORCE_INLINE
    static void singlerun(DataWorkspace<T>& p, const ALGO& info)
    {
        size_t nz = p.m_z.size();
        uint32  *ri = p.m_r.begin();
        uint32  *re = ri+nz;
        const T *zi = p.m_z.begin();

        for (; ri != re; ++zi, ++ri) {
            *ri = run(info, *zi);
            if (Check && !ok(*ri, &p.m_x[0], *zi)) {
                *ri = run<ALGO>(info, *zi); // repeat calculations (conveninent for debugging)
                error(*ri, std::distance(p.m_r.begin(),ri), &p.m_x[0], &p.m_z[0]);
            }
        }
    }

};


/*
   TEST DISPATCHING
*/

template <typename T, InstrSet I, Algos A>
struct Run4Traits
{
    typedef Info<I, T, A> algo_t;
};

template <typename T, Algos A>
struct Run4Traits<T,Scalar,A>
{
    typedef InfoScalar<T, A> algo_t;
};

// bridge with templates
template <typename T, InstrSet I, Algos A>
double run4(DataWorkspace<T>& ws)
{
    typedef typename Run4Traits<T, I, A>::algo_t algo_t;
    
    algo_t info(&ws.m_x[0], ws.m_x.size());
    
    Tester<T, I>::template singlerun<true>(ws, info);
    clock_t t1 = std::clock();
    for (size_t t = 0; t < nRepeat; ++t)
        Tester<T, I>::template singlerun<false>(ws, info);
    clock_t t2 = std::clock();
    ws.checkAndReset(A,Scalar);
    return (nRepeat * ws.m_z.size()) / (static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
}


#define ALGO_CASE(a)  case a: return run4<T, Instr, a>(ws);

template <typename T, InstrSet Instr>
double run3(Algos a, DataWorkspace<T>& ws)
{
    switch(a) {
        ALGO_CASE(Classic);
        ALGO_CASE(ClassicMod);
        ALGO_CASE(ClassicOffset);
        ALGO_CASE(BitSet);
        ALGO_CASE(BitSetNoPad);
        ALGO_CASE(Eytzinger);
#ifdef USE_MKL
        ALGO_CASE(MKL);
#endif
        ALGO_CASE(Direct);
        ALGO_CASE(Direct2);
        ALGO_CASE(DirectCache);
        default:
            throw "invalid algo";
    };
}


template <Precision P>
double run2(InstrSet i, Algos a, DataWorkspace<typename PrecTraits<P>::type>& ws)
{
    typedef typename PrecTraits<P>::type T;
    switch(i) {
        case Scalar:
            return run3<T,Scalar>(a, ws);
        case SSE:
            return run3<T,SSE>(a, ws);
#ifdef USE_AVX
        case AVX:
            return run3<T,AVX>(a, ws);
#endif
        default:
            throw "invalid instruction set";
    };
}

template <Precision P>
struct RunThroughput {
    static void run1(size_t precIndex, size_t nx, throughput_t& throughPut)
    {
        typedef typename PrecTraits<P>::type T;

        srand(15879);

        // run all tests
        for (size_t i = 0; i < nAvg; ++i) {

            DataWorkspace<T> ws( T(INTMIN), T(INTMAX), VecScope[nx] );

            for (size_t algoIndex = 0; algoIndex < nAlgo; ++algoIndex) {
                for (size_t instrIndex = 0; instrIndex < nInstr; ++instrIndex) {
                    double dt = run2<P>(InstrScope[instrIndex], AlgoScope[algoIndex], ws);
                    throughPut[precIndex][i][instrIndex][algoIndex] = dt;
                    if (i > 0)
                        throughPut[precIndex][nAvg][instrIndex][algoIndex] += dt / nAvg;
                    else
                        throughPut[precIndex][nAvg][instrIndex][algoIndex] = dt / nAvg;
                }
            }
        }

    }
};

template <Precision P>
struct RunSetup {
    static void run1(size_t precIndex, size_t nx, setup_t& setupcost)
    {
        typedef typename PrecTraits<P>::type T;

        srand(15879);

        SetupStats& stats = setupcost[P][0];
        stats.vec.resize(nAvgSetup);

        const size_t nr = nSetupTargetOps / (VecScope[nx]+1);
        cout << "\t" << P << " precision " << nr << " repetitions ... ";

        // run all tests
        for (size_t i = 0; i < nAvgSetup; ++i) {

            DataWorkspace<T> ws( T(INTMIN), T(INTMAX), VecScope[nx] );

            clock_t t1 = std::clock();
            for (size_t j = 0; j < nr; ++j)
                Info<SSE,T,Direct> info(ws.m_x);
            clock_t t2 = std::clock();
            double dt = ((static_cast<double>(t2 - t1) / nr) / CLOCKS_PER_SEC);
            if (dt==0) {
                cout << "zero found ";
            }
            stats.vec[i] = dt;
        }

        stats.computeStats();

        cout << nr << " done\n";
    }
};

template <Precision P>
struct RunR {
    static void run1(size_t precIndex, size_t nx, setup_t& setupcost)
    {
        typedef typename PrecTraits<P>::type T;

        srand(15879);

        SetupStats& statsInc = setupcost[P][0];
        SetupStats& statsRatio = setupcost[P][1];
        statsInc.vec.resize(nAvgSetupR);
        statsRatio.vec.resize(nAvgSetupR);

        cout << "\t" << P << " precision ... ";

        // run all tests
        for (size_t i = 0; i < nAvgSetupR; ++i) {
            DataWorkspace<T> ws(T(INTMIN), T(INTMAX), VecScope[nx]);
            Info<SSE, T, Direct> info(&ws.m_x[0], ws.m_x.size());
            statsInc.vec[i] = static_cast<double>(info.nInc);
            statsRatio.vec[i] = static_cast<double>(info.hRatio);
        }

        statsInc.computeStats();
        statsRatio.computeStats();

        cout << "done\n";
    }
};

template <template <Precision P> class Runner, class Result>
void run0(size_t nx, Result& result)
{

    for (size_t precIndex = 0; precIndex < nPrec; ++precIndex)
        switch(PrecScope[precIndex]) {
            case Single:
                Runner<Single>::run1(precIndex, nx, result);
                break;
            case Double:
                Runner<Double>::run1(precIndex, nx, result);
                break;
            default:
                throw "invalid precision";
        };
}


// declaration of output functions
void latex(size_t nx, const throughput_t& throughPut);
void latex(const setup_t results[], int which, const char *legend, double scaler, const char *fmt, bool normalize);
void print0(size_t nx, const throughput_t& throughPut);
void print0(const setup_t results[], int which, double scaler, const char *fmt, bool normalize);

int main(int argc, char* argv[])
{
    if (argc>1) {
        char c;
        cout << "press a key...";
        cin >> c;
    }

    cout << "Be patient, this test will take a few hours\n\n";

#if 1
    // throughput tests
    cout << "Testing throughput\n";
    for (size_t i = 0; i < nVecs; ++i) {
        throughput_t results;
        run0<RunThroughput>(i, results);
        print0(i, results);
     //   latex(i, results);
    }
#endif

#if 1
    // Direct methods setup cost tests
    setup_t results[nVecs];
    cout << "Testing setup cost\n";
    for (size_t i = 0; i < nVecs; ++i) {
        cout << "Testing with size " << VecScope[i] << "\n";
        run0<RunSetup>(i,results[i]);
    }
    print0(results, 0, 1.0e9, "%.3f", true);
    latex(results, 0, "Statistical setup cost for %s in nano seconds normalized by the array size", 1.0e9, "%.3f", true);
#endif

#if 0
    // Direct method check number of H grow
    setup_t resultsR[nVecs];
    cout << "Testing H growth\n";
    for (size_t i = 0; i < nVecs; ++i) {
        cout << "Testing with size " << VecScope[i] << "\n";
        run0<RunR>(i, resultsR[i]);
    }
    print0(resultsR, 0, 1.0, "%.4f", false);
    latex(resultsR, 0, "Number of $H$ updates", 1.0, "%.4f", false);
    print0(resultsR, 1, 1.0, "%.4f%%", false);
    latex(resultsR, 1, "Percentual growth of $H$ with respect to the predicted size", 100.0, "%.4f%%", false);
#endif

    return 0;
}

