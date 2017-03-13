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

//size_t rMin, rMax;

template <Precision P, Algos A, InstrSet I>
struct Tester
{
    typedef typename PrecTraits<P>::type T;
    typedef Info<P,A> info_t;
    typedef ExprVector<P,A,I> expr_t;

    static double run(DataWorkspace<P>& p,const info_t& info)
    {
        clock_t t1 = std::clock();
        for (size_t t = 0; t < nRepeat; ++t)
            singlerun(p,info);
        clock_t t2 = std::clock();
        p.checkAndReset(A,I);
        return (nRepeat * p.m_z.size()) / (static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
    }
private:
    NO_INLINE
    static void singlerun(DataWorkspace<P>& p, const info_t& info)
    {
        expr_t e;
        e.initN(p, info);
        Loop< expr_t >::loop(e, static_cast<uint32>( p.m_z.size() ));
    }
};


template <Precision P, Algos A>
struct Tester<P, A, Scalar>
{
    typedef typename PrecTraits<P>::type T;
    typedef Info<P,A> info_t;
    typedef ExprScalar<P,A> expr_t;

    static double run(DataWorkspace<P>& p, const info_t& info )
    {
        size_t nz = p.m_z.size();
        clock_t t1 = std::clock();
        for (size_t t = 0; t < nRepeat; ++t) {
            for (uint32 j = 0; j < nz; ++j)
                singlerun(p, info, j);
        }
        clock_t t2 = std::clock();
        p.checkAndReset(A,Scalar);
        return (nRepeat * nz) / (static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
    }

private:
    NO_INLINE
    static void singlerun(DataWorkspace<P>& p, const info_t& info, uint32 j)
    {
#if defined(DEBUG)
        int redo = 0;
        do {
#endif
            expr_t e;
            e.init0(p, info);
            e.scalar(j);
#if defined(DEBUG)
            uint32 ri = p.m_r[j];
            T zi = p.m_z[j];
            if (!ok(ri, &p.m_x[0], zi)) {
                ++redo;
                error(ri, j, &p.m_x[0], &p.m_z[0]);
            }
        } while (redo == 1);
#endif
    }
};


/*
   TEST DISPATCHING
*/

// bridge with templates
template <Precision P, InstrSet I, Algos A>
double run4(DataWorkspace<P>& ws)
{
    Info<P,A> info(ws.m_x);
    return Tester<P, A, I>::run(ws, info);
}


#define ALGO_CASE(a)  case a: return run4<P, Instr, a>(ws);

template <Precision P, InstrSet Instr>
double run3(Algos a, DataWorkspace<P>& ws)
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
double run2(InstrSet i, Algos a, DataWorkspace<P>& ws)
{
    switch(i) {
        case Scalar:
            return run3<P,Scalar>(a, ws);
        case SSE:
            return run3<P,SSE>(a, ws);
#ifdef USE_AVX
        case AVX:
            return run3<P,AVX>(a, ws);
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

            DataWorkspace<P> ws( T(INTMIN), T(INTMAX), VecScope[nx] );

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

            DataWorkspace<P> ws( T(INTMIN), T(INTMAX), VecScope[nx] );

            clock_t t1 = std::clock();
            for (size_t j = 0; j < nr; ++j)
                Info<P,Direct> info(ws.m_x);
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
            DataWorkspace<P> ws(T(INTMIN), T(INTMAX), VecScope[nx]);
            Info<P, Direct> info(ws.m_x);
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

#if 1
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

