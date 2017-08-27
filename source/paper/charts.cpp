#include <iostream>
#include <cstring>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>

#include "Output.h"
#include "TypesForArticleTest.h"

using namespace std;

void parseValue(const std::string& cell, std::string& result)
{
    result = cell;
}

void parseValue(const std::string& cell, double& result)
{
    result = std::stod(cell);
}

void parseValue(const std::string& cell, size_t& result)
{
    result = std::stoul(cell);
}

template <typename T>
T readtoken(std::istringstream& is)
{
    std::string        cell;
    if (!std::getline(is, cell, ';')) {
        std::cout << "Error parsing input file\n";
        exit(1);
    }

    T result;
    parseValue(cell, result);
    return result;
}

struct SeriesCmp {
    bool operator()(Algos a, Algos b) const
    {
        // very inefficient, but who cares? It is just for generation of the document charts.
        int x = SeriesOrder.find(a)->second;
        int y = SeriesOrder.find(b)->second;
        return x < y;
    }
};

typedef std::pair<size_t, double> point_t;
typedef std::vector<point_t> series_t;
typedef std::map<Algos, series_t, SeriesCmp> plot_t;

void getchartseries(const std::string& inpfilename, InstrSet instr, Precision prec, plot_t& plot)
{
    typedef std::map<string, series_t> strplot_t;

    strplot_t strplot;

    std::ifstream inpf(inpfilename);
    if (!inpf) {
        cout << "Error opening file: " << inpfilename << "\n";
        exit(1);
    }

    std::string lpname = PrecNames[prec];
    std::string liname = InstrNames[instr];

    std::string line;
    if (!std::getline(inpf, line))  // ignore headers
        exit(1);
    else
        cout << "ignoring line: " << line << "\n";
    while (std::getline(inpf, line))
    {
        cout << "processing line: " << line << "\n";
        std::istringstream  lineStream(line);
        size_t vec = readtoken<size_t>(lineStream);
        std::string a = readtoken<std::string>(lineStream);
        std::string p = readtoken<std::string>(lineStream);
        std::string i = readtoken<std::string>(lineStream);
        double dt = readtoken<double>(lineStream);
        if (p == lpname && i == liname)
            strplot[a].push_back(point_t(vec, dt));
        else
            cout << "\tignore: " << p << "!=" << lpname << " OR " << i << "!=" << liname << "\n";
    }

    // reorder plot
    for (strplot_t::const_iterator aiter = strplot.begin(); aiter != strplot.end(); ++aiter)
        plot[algoCodeFromName(aiter->first)] = aiter->second;
}

void chartseries2latex(const std::string& ofilenameprefix, const plot_t& plot, InstrSet instr, Precision prec, bool addLegend)
{
    std::string spname = ChartPrecNames[prec];
    std::string siname = ChartInstrNames[instr];

    std::ofstream of(ofilenameprefix + siname + "-" + spname + ".perfplot");

    std::string legend = "\\legend{";
    for (plot_t::const_iterator aiter = plot.begin(); aiter != plot.end(); ++aiter) {
        Algos algo = aiter->first;
        string name = AlgoNames[algo];
        /*
        if (name=="Ternary")
        name += "(SSE)";
        if (name=="Nonary")
        name += "(AVX)";
        if (name == "Pentary")
        name += (prec==Single)? "(SSE)": "(AVX)";
        */
        const series_t& series = aiter->second;

        of << "\\addplot[\n"
            << "\tcolor=" << color(name) << ",\n"
            << "\tmark=" << mark(name) << ",\n"
            << "\tstyle=" << style(name) << ",\n"
            << "]\n"
            << "coordinates {\n";
        for (series_t::const_iterator piter = series.begin(); piter != series.end(); ++piter)
            of << "(" << piter->first << "," << piter->second << ")";
        of << "\n};\n";
        if (aiter != plot.begin())
            legend += ",";
        legend += texname(name);
    }
    legend += "}";
    if (addLegend)
        of << legend << "\n";
}


void makechartseries(const std::string& ifilename, const std::string& ofilenameprefix, InstrSet instr, Precision prec)
{
    plot_t plot;
    getchartseries(ifilename, instr, prec, plot);
    chartseries2latex(ofilenameprefix, plot, instr, prec, true);
}

void makechartseries(const std::string& ifilename, const std::string& ofilenameprefix, InstrSet instr)
{
    plot_t plots, plotd;
    getchartseries(ifilename, instr, Single, plots);
    getchartseries(ifilename, instr, Double, plotd);

    // very inefficient loops, but the size is small
    vector<point_t> dummyvec(1, point_t(1, 0.1)); // dummy point which will not be rendered
    for (plot_t::const_iterator siter = plots.begin(); siter != plots.end(); ++siter)
        plotd.insert(make_pair(siter->first, dummyvec));  // only insert if not already present
    for (plot_t::const_iterator diter = plotd.begin(); diter != plotd.end(); ++diter)
        plots.insert(make_pair(diter->first, dummyvec));  // only insert if not already present

    chartseries2latex(ofilenameprefix, plots, instr, Single, false);
    chartseries2latex(ofilenameprefix, plotd, instr, Double, true);
}


int main(int argc, char* argv[])
{
    populateNames();

    std::string inputfile(argv[1]);
    std::string outputfileprefix(argc>2?argv[2]:"");

/*
    makechartseries(fn, Scalar, Single);
    makechartseries(fn, Scalar, Double);
    makechartseries(fn, SSE, Single);
    makechartseries(fn, SSE, Double);
    makechartseries(fn, AVX, Single);
    makechartseries(fn, AVX, Double);
*/
    makechartseries(inputfile, outputfileprefix, Scalar);
    makechartseries(inputfile, outputfileprefix, SSE);
    makechartseries(inputfile, outputfileprefix, AVX);
}
