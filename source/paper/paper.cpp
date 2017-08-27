#include <iostream>
#include <cstring>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <set>
#include <cstdio>
#include <ctime>
#include <sstream>
#include <limits>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <string>
#include <iomanip>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Output.h"
#include "SIMD.h"
#include "Workspace.h"

// algorithms
#include "Algo-Classic.h"
#include "Algo-ClassicMod.h"
#include "Algo-ClassicOffset.h"
#include "Algo-MorinOffset.h"
#include "Algo-LowerBound.h"
#include "Algo-MorinBranchy.h"
#include "Algo-BitSet.h"
#include "Algo-BitSetNoPad.h"
#include "Algo-MKL.h"
#include "Algo-Direct.h"
#include "Algo-Direct2.h"
#include "Algo-DirectCache.h"
#include "Algo-Eytzinger.h"
#include "Algo-KAry.h"

using namespace BinSearch;


// generate result table in latex format
void latex(size_t nx, const throughput_t& throughPut)
{
    // table definition
    cout << "\\begin{table}[ht]" << endl;
    cout << "\\centering\n\\footnotesize\n";
    cout << "\\begin{tabular}{l |";
    for (size_t p = 0; p < nPrec; ++p ) {
        for (size_t i = 0; i < nInstr; ++i )
            cout << " c";
        cout << " |";
    }
    cout << "}" << endl;

    // partial horizontal line
    hpartialline(2, 1 + nPrec*nInstr);

    // main header
    cout << "             ";
    for (size_t p = 0; p < nPrec; ++p ) {
         cout << " & \\multicolumn{" << nInstr << "}{c|}{\\textbf{" << PrecNames[PrecScope[p]] << "}}";
    }
    cout << texeol;

    // partial horizontal line
    hpartialline(2, 1 + nPrec*nInstr);

    // instruction set headers row
    cout << "             ";
    for (size_t p = 0; p < nPrec; ++p )
        for (size_t i = 0; i < nInstr; ++i )
            cout << " & \\textbf{" << InstrNames[InstrScope[i]] << "}";
    cout << texeol;
    cout << "             ";
    for (size_t p = 0; p < nPrec; ++p)
        for (size_t i = 0; i < nInstr; ++i)
            cout << " & $\\mathbf{d=" << getD(PrecScope[p], InstrScope[i]) << "}$";
    cout << texeol;

    // horizontal line
    cout << "\\hline" << endl;

    // test results
    for (size_t a = 0; a < nAlgo; ++a ) {
        ostringstream os;
        os << "{" << texname(AlgoNames[AlgoScope[a]]) << "}";
        cout << "\\multicolumn{1}{|c|}{\\textbf" << left << setw(algoTexSize) << os.str() << "}" << right;
        for (size_t p = 0; p < nPrec; ++p ) {
            for (size_t i = 0; i < nInstr; ++i ) {
                double dt = throughPut[p][nAvg][i][a] / scaler;
                cout << " & ";
                cout << checkAvail(dt);
            }
        }
        cout << texeol;
    }

    cout << "\\hline" << endl;
    cout << "\\end{tabular}" << endl;
    cout << "\\caption{Throughput in millions of searches per second with vector $X$ of size " << VecScope[nx] << "}" << endl;
    cout << "\\label{tab:results" << nx << "}" << endl;
    cout << "\\end{table}" << endl;

    cout << endl;
}
// print results to screen
void print0(size_t nx, const throughput_t& throughPut)
{
    cout << "Results with NX = " << VecScope[nx] << endl;
    //std::cout << "R-range: " << rMin << "-" << rMax << "\n";

    // precision header row
    cout << setw(nameSize) << "";
    for (size_t p = 0; p < nPrec; ++p )
        cout << setw(precSize) << center(PrecNames[PrecScope[p]], precSize);
    cout << endl;


    // instr header rows
    cout << setw(nameSize) << "";
    for (size_t p = 0; p < nPrec; ++p )
        for (size_t i = 0; i < nInstr; ++i )
            cout << setw(instrSize) << InstrNames[InstrScope[i]];
    cout << endl;

    // instr/prec d rows
    cout << setw(nameSize) << "";
    for (size_t p = 0; p < nPrec; ++p )
        for (size_t i = 0; i < nInstr; ++i ) {
            ostringstream os;
            os << "d=" << getD(PrecScope[p], InstrScope[i]);
            cout << setw(instrSize) << os.str();
        }
    cout << endl;

    // algos rows
    for (size_t a = 0; a < nAlgo; ++a ) {
        cout << setw(nameSize) << AlgoNames[AlgoScope[a]];
        for (size_t p = 0; p < nPrec; ++p )
            for (size_t i = 0; i < nInstr; ++i ) {
                double dt = throughPut[p][nAvg][i][a] / scaler;
                cout << checkAvail(dt);
            }
        cout << endl;
    }

    cout << endl;
}

void print0(const setup_t results[], int which, double f, const char *fmt, bool normalize)
{
    // precision header row
    cout << std::left << setw(nameSize) << "Array Size";
    for (size_t p = 0; p < nPrec; ++p )
        cout << std::left << setw(4*instrSize) << PrecNames[PrecScope[p]];
    cout << endl;

    cout << setw(nameSize) << "";
    for (size_t p = 0; p < nPrec; ++p ) {
        cout << setw(instrSize) << "mean";
        cout << setw(instrSize) << "min";
        cout << setw(instrSize) << "max";
        cout << setw(instrSize) << "stdev";
    }
    cout << endl;

    char buffer[32];
    for (size_t i = 0; i < nVecs; ++i) {
        cout << setw(nameSize) << VecScope[i];
        const double factorV = normalize? f/VecScope[i] : f;
        for (size_t p = 0; p < nPrec; ++p ) {
            cout << setw(instrSize) << format(buffer, fmt, results[i][p][which].avg * factorV);  // micro seconds
            cout << setw(instrSize) << format(buffer, fmt, results[i][p][which].mi * factorV);  // micro seconds
            cout << setw(instrSize) << format(buffer, fmt, results[i][p][which].ma * factorV);  // micro seconds
            cout << setw(instrSize) << format(buffer, fmt, results[i][p][which].stdev * factorV);  // micro seconds
        }
        cout << endl;
    }

    cout << endl;
}

// generate result table in latex format
void latex(const setup_t results[], int which, const char *formatmsg, double f, const char *fmt, bool normalize)
{
    const size_t ncols = 4;
    const char *outputCols[ncols] = { "mean", "min", "max", "stdev"};

    // table definition
    cout << "\\begin{table}[h]" << endl;
    cout << "\\begin{tabular}{| c |";
    for (size_t p = 0; p < nPrec; ++p) {
        for (size_t i = 0; i < ncols; ++i)
            cout << " c";
        cout << " |";
    }
    cout << "}" << endl;

    // partial horizontal line
    hpartialline(2, 1 + ncols*nPrec);

    // main header
    cout << " \\multicolumn{1} {c|}{}";
    for (size_t p = 0; p < nPrec; ++p) {
        cout << "  & \\multicolumn{" << ncols << "}{c|}{\\textbf{" << PrecNames[PrecScope[p]] << "}}";
    }
    cout << texeol;

    // horizontal line
    cout << "\\hline" << endl;

    // output value headers row
    cout << "\\textbf{array size}";
    //cout << "\\multicolumn{1}{|c|}{\\textbf" << left << setw(algoTexSize) << "array size" << "}" << right;
    for (size_t p = 0; p < nPrec; ++p) {
        for (size_t i = 0; i < ncols; ++i) {
            cout << " & \\textbf{" <<outputCols[i] << "}";
        }
    }
    cout << texeol;

    // horizontal line
    cout << "\\hline" << endl;

    char buffer[1000];

    // test results
    for (size_t v = 0; v < nVecs; ++v) {
        ostringstream os;
        os << "{" << VecScope[v] << "}";
        cout << "\\multicolumn{1}{|c|}{\\textbf" << left << setw(algoTexSize) << os.str() << "}" << right;
        const double factorV = normalize ? f / VecScope[v] : f;
        for (size_t p = 0; p < nPrec; ++p) {
                cout << " & " << setw(instrSize) << format(buffer, fmt, results[v][p][which].avg * factorV);
                cout << " & " << setw(instrSize) << format(buffer, fmt, results[v][p][which].mi  * factorV);
                cout << " & " << setw(instrSize) << format(buffer, fmt, results[v][p][which].ma  * factorV);
                cout << " & " << setw(instrSize) << format(buffer, fmt, results[v][p][which].stdev * factorV);
        }
        cout << texeol;
    }

    cout << "\\hline" << endl;
    cout << "\\end{tabular}" << endl;
    sprintf(buffer, formatmsg, texname(AlgoNames[Direct]).c_str());
    cout << "\\caption{" << buffer << "}" << endl;
    cout << "\\label{tab:results" << 100 << "}" << endl;
    cout << "\\end{table}" << endl;

    cout << endl;
}

void tabulateperformance(const char *filename, const throughput_t* perfresults)
{
    std::ofstream of(filename);
    of << "vecsize;algorithm;precision;instruction;time\n";
    for (size_t v = 0; v < nVecs; ++v) {
        for (size_t a = 0; a < nAlgo; ++a ) {
            for (size_t p = 0; p < nPrec; ++p ) {
                for (size_t i = 0; i < nInstr; ++i ) {
                    double dt = perfresults[v][p][nAvg][i][a] / scaler;
                    if(!std::isnan(dt)) {
                        of  << VecScope[v] << ";"
                            << AlgoNames[AlgoScope[a]] << ";"
                            << PrecNames[PrecScope[p]] << ";"
                            << InstrNames[InstrScope[i]] << ";"
                            << dt << "\n";
                    }
                }
            }
        }
    }
}

/*
   Algo wrappers
*/

template <typename T, Algos A, bool Complete = Details::IsComplete<Details::AlgoScalarBase<T, A>>::value>
struct AlgoScalarTester;

template <typename T, Algos A>
struct AlgoScalarTester<T,A,true> : public Details::AlgoScalarBase<T,A>
{
    typedef Details::AlgoScalarBase<T, A> base_t;

    AlgoScalarTester(const T* px, const uint32 n) :  base_t(px, n) {}
};

template <InstrSet I, typename T, Algos A, bool Complete=Details::IsComplete<Details::AlgoVecBase<I, T, A>>::value>
struct AlgoVecTester;

template <InstrSet I, typename T, Algos A>
struct AlgoVecTester<I,T,A,true> : Details::AlgoVecBase<I, T, A>
{
    typedef Details::AlgoVecBase<I, T, A> base_t;

    AlgoVecTester(const T* px, const uint32 n) :  base_t(px, n) {}

    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz, uint32 n) const
    {
        Details::Loop<T,base_t>::loop(*this, pr, pz, n);
    }
};

#ifdef USE_MKL

template <InstrSet I, typename T>
struct AlgoVecTester<I,T,MKL,false> : Details::AlgoVecMKL<T>
{
    typedef Details::AlgoVecMKL<T> base_t;

    AlgoVecTester(const T* px, const uint32 n) :  base_t(px, n) {}

    FORCE_INLINE
    void vectorial(uint32 *pr, const T *pz, uint32 n) const
    {
        base_t::vectorial(pr, pz, n);
    }
};

#endif


/*
    The following calss are designed to make the total computation cost roughly comparable
    for the various algorithms
*/

template <InstrSet I, Algos A>
struct CostTraits
{
    static size_t get(uint32 vecSize, size_t nr)
    {
        return 2 * CostTraits<Scalar, A>::get(vecSize,nr);
    }
};

template <Algos A>
struct CostTraits<Scalar,A>
{
    static size_t get(uint32 vecSize, size_t nr)
    {
        double n = ceil(log2(vecSize)) / 4.0;
        size_t r = std::max<size_t>( 4, static_cast<size_t>(nr / n) );
        //std::cout << r << "\n";
        return r;
    }
};


template <typename T, InstrSet I>
class Tester
{
public:
    template <bool Check, typename ALGO>
    NO_INLINE
    static void singlerun(DataWorkspace<T>& p, const ALGO& algo)
    {
        uint32 n = static_cast<uint32>(p.m_z.size());
        algo.vectorial(p.m_r.begin(), p.m_z.begin(), n);
    }
};


template <typename T>
class Tester<T, Scalar>
{
    template <typename ALGO>
    NO_INLINE
    static uint32 run(const ALGO& algo, T zi)
    {
        return algo.scalar(zi);
    }

public:
    template <bool Check, typename ALGO>
    FORCE_INLINE
    static void singlerun(DataWorkspace<T>& p, const ALGO& algo)
    {
        size_t nz = p.m_z.size();
        uint32  *ri = p.m_r.begin();
        uint32  *re = ri+nz;
        const T *zi = p.m_z.begin();
        uint32 nErr;

        if (Check)
            nErr = 0;

        for (; ri != re; ++zi, ++ri) {
            *ri = run(algo, *zi);
            if (Check && !ok(*ri, &p.m_x[0], *zi) && nErr < 4) {
                *ri = run<ALGO>(algo, *zi); // repeat calculations (convenient for debugging)
                error(*ri, std::distance(p.m_r.begin(),ri), &p.m_x[0], &p.m_z[0]);
                ++nErr;
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
    typedef AlgoVecTester<I, T, A> algo_t;
};

template <typename T, Algos A>
struct Run4Traits<T,Scalar,A>
{
    typedef AlgoScalarTester<T, A> algo_t;
};

// bridge with templates
template <typename T, InstrSet I, Algos A>
typename enable_if<Details::IsComplete<typename Run4Traits<T, I, A>::algo_t>::value, double>::type run4(DataWorkspace<T>& ws)
{
    typedef typename Run4Traits<T, I, A>::algo_t algo_t;

    uint32 vecsize = static_cast<uint32>(ws.m_x.size());
    algo_t algo(&ws.m_x[0], vecsize);

    Tester<T, I>::template singlerun<true>(ws, algo);
    size_t nr = CostTraits<I,A>::get(vecsize, nRepeat);
#ifdef _OPENMP
    size_t nt = std::stoul(std::getenv("OMP_NUM_THREADS"));
    //std::cout << nt << "\n";
    nr *= nt;
#endif
    clock_t t1 = std::clock();
#ifdef _OPENMP
#   pragma omp parallel for
#endif
    for (size_t t = 0; t < nr; ++t)
        Tester<T, I>::template singlerun<false>(ws, algo);
    clock_t t2 = std::clock();
    ws.checkAndReset(A,I);
    return (static_cast<double>(nr) * ws.m_z.size()) / (static_cast<double>(t2 - t1) / CLOCKS_PER_SEC);
}

template <typename T, InstrSet I, Algos A>
typename enable_if<!Details::IsComplete<typename Run4Traits<T, I, A>::algo_t>::value, double>::type run4(DataWorkspace<T>& ws)
{
    return std::numeric_limits<double>::quiet_NaN();
}

#define ALGO_CASE(a)  case a: return run4<T, Instr, a>(ws);

template <typename T, InstrSet Instr>
double run3(Algos a, DataWorkspace<T>& ws)
{
    switch(a) {
#define ALGOENUM(a,b) case a: return run4<T, Instr, a>(ws);
#include "AlgoXCodes.h"
#undef ALGOENUM
        default:
            throw "invalid algo";
    };
}


template <typename T>
double run2(InstrSet i, Algos a, DataWorkspace<T>& ws)
{
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
                    double dt = run2<T>(InstrScope[instrIndex], AlgoScope[algoIndex], ws);
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
                AlgoVecTester<SSE,T,Direct> algo(&ws.m_x[0], ws.m_x.size());
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
            AlgoVecTester<SSE, T, Direct> algo(&ws.m_x[0], ws.m_x.size());
            statsInc.vec[i] = static_cast<double>(algo.nInc);
            statsRatio.vec[i] = static_cast<double>(algo.hRatio);
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
void populateNames();
void latex(size_t nx, const throughput_t& throughPut);
void latex(const setup_t results[], int which, const char *legend, double scaler, const char *fmt, bool normalize);
void print0(size_t nx, const throughput_t& throughPut);
void print0(const setup_t results[], int which, double scaler, const char *fmt, bool normalize);
void tabulateperformance(const char *filename, const throughput_t* perfresults);

int main(int argc, char* argv[])
{
#if defined(USE_MKL) && defined(__CYGWIN__)
    InitMKLWrapper();
#endif

    if (argc>1) {
        char c;
        cout << "press a key...";
        cin >> c;
    }

    cout << "Be patient, this test will take a few hours\n\n";

    populateNames();

#if 1
    // throughput tests
    throughput_t perfresults[nVecs];
    cout << "Testing throughput\n";
    for (size_t i = 0; i < nVecs; ++i) {
        run0<RunThroughput>(i, perfresults[i]);
        print0(i, perfresults[i]);
        // latex(i, perfresults[i]);
    }
    tabulateperformance("perf.csv", perfresults);

#endif

#if 0
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

#if defined(USE_MKL) && defined(__CYGWIN__)
    ReleaseMKLWrapper();
#endif

    return 0;
}


#if defined(USE_MKL) && defined(__CYGWIN__)

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>

#include "mklloader.h"

dfdNewTask1D_t           dfdNewTask1D_ptr;
dfsNewTask1D_t           dfsNewTask1D_ptr;
dfdSearchCells1D_t       dfdSearchCells1D_ptr;
dfsSearchCells1D_t       dfsSearchCells1D_ptr;
dfDeleteTask_t           dfDeleteTask_ptr;

HINSTANCE hDLL;               // Handle to DLL

template <typename T>
void getAddress(T& pf, const char *name)
{
    pf = (T)GetProcAddress(hDLL, name);
    if (!pf) {
        cout << "Error opntaining function pointer for: " << name << "\n";
        FreeLibrary(hDLL);
        exit(1);
    }
}

void InitMKLWrapper()
{
    hDLL = LoadLibrary("mklwrap.dll");

    if (hDLL != NULL) {
        mysum_t testsum;
        getAddress(testsum, "mysum_wrap");
        if ((*testsum)(10, 12) != 22) {
            cout << "Error with functions signatures\n";
            exit(1);
        }
        getAddress(dfdNewTask1D_ptr, "dfdNewTask1D_wrap");
        getAddress(dfsNewTask1D_ptr, "dfsNewTask1D_wrap");
        getAddress(dfdSearchCells1D_ptr, "dfdSearchCells1D_wrap");
        getAddress(dfsSearchCells1D_ptr, "dfsSearchCells1D_wrap");
        getAddress(dfDeleteTask_ptr, "dfDeleteTask_wrap");
    }
    else {
        cout << "Error loading MKL wrapper library\n";
        exit(1);
    }
    cout << "MKL pointers initialized\n";
}

void ReleaseMKLWrapper()
{
    FreeLibrary(hDLL);
}

#endif
