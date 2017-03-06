#include <iostream>
#include <cstring>
#include <string>
#include <sstream>
#include <iomanip>

#include "Type.h"
#include "Config.h"
#include "Unroller.h"

using namespace std;

//extern size_t rMin, rMax;

const char *PrecNames[] = { "Single Precision", "Double Precision" };
const char *InstrNames[] = { "Scalar", "SSE-4", "AVX-2" };
const char *AlgoNames[] = { "Classic", "ClassicMod", "ClassicOffset", "BitSet", "BitSetNoPad", "Eytzinger", "MKL", "Direct", "Direct-2", "DirectCache" };
const char *texName[] =
    {
        "Algorithm \\ref{alg:naivealg}",
        "Algorithm \\ref{alg:naivemodalg}",
        "Algorithm \\ref{alg:naiveoffset}",
        "Algorithm \\ref{alg:binaryopt}",
        "Algorithm \\ref{alg:binaryoptnopad}",
        "Algorithm \\ref{alg:eytzinger}",
        "MKL 2017",
        "Algorithm \\ref{alg:direct}",
        "Algorithm \\ref{alg:direct-2}",
        "Algorithm \\ref{alg:directcache}",
    };

inline size_t getD( Precision p, InstrSet i)
{
    if (i == Scalar)
        return 1;

    size_t s1 = (p == Single)? 4: 8;
    switch(i) {
        case SSE:
            return 16/s1;
        case AVX:
            return 32/s1;
        default:
            throw "invalid instruction set";
    };

}

// these constants control the formatting of the output
const size_t instrSize = 10;
const size_t nameSize = 15;
const size_t precSize = instrSize*nInstr;
const size_t algoTexSize = 40;
const int ndecimals = 2;
const double scaler = 1e6;
const double setupscaler = 1e9;
const string texeol(" \\\\\n");

void hpartialline(size_t col1, size_t col2)
{
    cout << "\\cline{" << col1 << "-" << col2 << "}" << endl;
}

// generate result table in latex format
void latex(size_t nx, const throughput_t& throughPut)
{
    std::cout << fixed << setprecision(ndecimals);

    // table definition
    cout << "\\begin{table}[h]" << endl;
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
    for (size_t p = 0; p < nPrec; ++p ) {
        for (size_t i = 0; i < nInstr; ++i ) {
            cout << " & \\testmode{" << InstrNames[InstrScope[i]] << "}{" << getD(PrecScope[p], InstrScope[i]) << "}";
        }
    }
    cout << texeol;

    // horizontal line
    cout << "\\hline" << endl;

    // test results
    for (size_t a = 0; a < nAlgo; ++a ) {
        ostringstream os;
        os << "{" << texName[AlgoScope[a]] << "}";
        cout << "\\multicolumn{1}{|c|}{\\textbf" << left << setw(algoTexSize) << os.str() << "}" << right;
        for (size_t p = 0; p < nPrec; ++p ) {
            for (size_t i = 0; i < nInstr; ++i ) {
                cout << " & " << setw(instrSize) << throughPut[p][nAvg][i][a] / scaler;
            }
        }
        cout << texeol;
    }

    cout << "\\hline" << endl;
    cout << "\\end{tabular}" << endl;
    cout << "\\caption{Throughput in million of searches per second with vector $X$ of size " << VecScope[nx] << "}" << endl;
    cout << "\\label{tab:results" << nx << "}" << endl;
    cout << "\\end{table}" << endl;

    cout << endl;
}

string center(const char *s, size_t l)
{
    size_t ls = strlen(s);
    size_t pad = l<=ls? 0 : (l-ls) / 2;
    ostringstream os;
    os << setw(pad) << "" << s << setw(pad) << "";
    return os.str();
}

// print results to screen
void print0(size_t nx, const throughput_t& throughPut)
{
    cout << "Results with NX = " << VecScope[nx] << endl;
    //std::cout << "R-range: " << rMin << "-" << rMax << "\n";
    std::cout << fixed << setprecision(ndecimals);

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
            for (size_t i = 0; i < nInstr; ++i )
                cout << setw(instrSize) << throughPut[p][nAvg][i][a] / scaler;
        cout << endl;
    }

    cout << endl;
}

const char *format(char *dest, const char *fmt, double v)
{
    sprintf(dest, fmt, v);
    return dest;
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
    sprintf(buffer, formatmsg, texName[Direct]);
    cout << "\\caption{" << buffer << "}" << endl;
    cout << "\\label{tab:results" << 100 << "}" << endl;
    cout << "\\end{table}" << endl;

    cout << endl;
}
