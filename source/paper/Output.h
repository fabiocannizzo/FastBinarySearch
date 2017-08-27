#pragma once

//#include <iostream>
//#include <cstring>
//#include <string>
//#include <sstream>
//#include <fstream>
//#include <iomanip>
#include <map>
//#include <set>

#include "TypesForArticleTest.h"
#include "Config.h"

using namespace std;

map<InstrSet, string> InstrNames;
map<InstrSet, string> ChartInstrNames;
map<Algos, string> AlgoNames;
map<Precision, string> PrecNames;
map<Precision, string> ChartPrecNames;
map<Algos, int> SeriesOrder;

void populateNames()
{
    PrecNames[Single] = "Single";
    PrecNames[Double] = "Double";

    ChartPrecNames[Single] = "single";
    ChartPrecNames[Double] = "double";

    InstrNames[Scalar]  = "Scalar";
    InstrNames[SSE]     = "SSE-4";
    InstrNames[AVX]     = "AVX-2";

    ChartInstrNames[Scalar]  = "scalar";
    ChartInstrNames[SSE]     = "SSE";
    ChartInstrNames[AVX]     = "AVX";

#define ALGOENUM(x, b) AlgoNames[x] = #x;
#include "AlgoXCodes.h"
#undef ALGOENUM

#define ALGOENUM(x, b) SeriesOrder[x] = b;
#include "AlgoXCodes.h"
#undef ALGOENUM
}

Algos algoCodeFromName(const std::string& name)
{
    for(map<Algos,string>::const_iterator i = AlgoNames.begin(); i != AlgoNames.end(); ++i)
       if(i->second == name)
           return i->first;
    cout << "Algo name not found: " << name << "\n";
    exit(1);
}

string texify(const string& name, const string& suffix)
{
    return ("\\" + name + suffix);
}

string texname(const string& name)
{
    return texify(name, "Name");
}

string color(const string& name)
{
    return ("\\" + name + "Color");
}

string style(const string& name)
{
    return ("\\" + name + "Style");
}

string mark(const string& name)
{
    return texify(name, "Mark");
}

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
const string notAvail = "---";

void hpartialline(size_t col1, size_t col2)
{
    cout << "\\cline{" << col1 << "-" << col2 << "}" << endl;
}

string checkAvail(double dt)
{
    std::ostringstream os;
    os << setw(instrSize);
    if (std::isnan(dt))
        os << "---";
    else
        os << fixed << setprecision(ndecimals) << dt;
    return os.str();
}

string center(const string& s, size_t l)
{
    size_t ls = s.length();
    size_t pad = l<=ls? 0 : (l-ls) / 2;
    ostringstream os;
    os << setw(pad) << "" << s << setw(pad) << "";
    return os.str();
}


const char *format(char *dest, const char *fmt, double v)
{
    sprintf(dest, fmt, v);
    return dest;
}

