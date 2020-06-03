#include "BinSearch.h"
#include <iostream>

using namespace BinSearch;
using namespace std;

double x[] = { 1, 2, 4, 5, 9 };
double z[] = { 1, 2, 4, 5, 1.5, 2.5, 4.8, 8.2 };

const uint32 n = sizeof(x)/sizeof(*x);
const uint32 m = sizeof(z) / sizeof(*z);

void displayResults(uint32 *j, const char *msg)
{
    cout << msg << " results:" << endl;
    for (size_t i = 0; i < m; ++i)
    {
        // show result
        cout << x[j[i]] << " <= " << z[i] << " < " << x[j[i] + 1] << endl;

        // check result
        if (!(x[j[i]] <= z[i] && z[i] < x[j[i] + 1]))
        {
            cout << "incorrect result!";
            exit(1);
        }

        // clear result
        j[i] = 0xFFFFFFFF;
    }
}

int main()
{
    uint32 j[m];

    // construct bin search algorithm
    BinAlgo<SSE, double, Ternary> bin_searcher(x, n);

    // search the bin for values in z using scalar search
    for (size_t i = 0; i < m; ++i)
        j[i] = bin_searcher.scalar(z[i]);

    // show and clear results
    displayResults(j, "scalar");

    // search the bin for values in z using vectorial search
    bin_searcher.vectorial(j, z, m);

    // show and clear results
    displayResults(j, "vectorial");

    return 0;
}
