#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>

#include "nagfbs.h"

using namespace std;

fbs_float_t x[] = { 1, 1.5, 2, 4, 5, 9 };
fbs_float_t z_[] = { 1, 2, 4, 5, 1.5, 2.5, 4.8, 8.2 };

const fbs_uint_t n = sizeof(x) / sizeof(*x);
const fbs_uint_t m = sizeof(z_) / sizeof(*z_);

template <bool L, bool R>
void displayResults(const fbs_float_t *z, fbs_uint_t m, fbs_uint_t *j, const char *msg)
{
    cout << msg << " results:" << endl;
    for (fbs_uint_t i = 0; i < m; ++i)
    {
        fbs_uint_t k = j[i];
        if ((!L && !R) || k < n) {

            // show result
            cout << x[k] << " <= " << z[i] << " < " << x[k + 1] << endl;

            // check result
            if (!(x[k] <= z[i] && z[i] < x[k + 1]))
            {
                cout << "incorrect result!";
                exit(1);
            }
        }
        else if (R && k == n) {
            //show result
            cout << x[n-1] << " <= " << z[i] << endl;

            // check result
            if (!(x[n-1] <= z[i]))
            {
                cout << "incorrect result!";
                exit(1);
            }
        }
        else if (L && k == numeric_limits<fbs_uint_t>::max()) {
            // show result
            cout << z[i] << " < " << x[0] << endl;

            // check result
            if (!(x[0] > z[i]))
            {
                cout << "incorrect result!";
                exit(1);
            }
        }
        else {
            // show result
            cout << "z=" << z[i] << ", k=" << k << endl;

            // check result
            cout << "incorrect result!\n";
            exit(1);
        }

        // clear result
        j[i] = 0xFFFFFFFF;
    }
}

void checked(int retVal, const char * msg)
{
    if (retVal) {
        std::cout << msg << "\n";
        exit(retVal);
    }
}

template <bool L, bool R>
void demo(FBS_Code code)
{
    cout << "Demo: " << code << "-" << L << "-" << R << endl;

    const fbs_uint_t mm = m+L+R;
    fbs_float_t zz[mm];
    copy(z_, z_+m, zz);
    if (L)
        zz[m] = x[0]-1;
    if (R)
        zz[m+L] = x[n-1]+1;

    int retVal;

    FBS_Info info;
    fbs_size_t mem;

    const bool left = L;
    const bool right = R;

    // STEP 1: get info
    retVal = FBS_getInfo(&code, x, &n, &left, &right, &mem, &info);
    checked(retVal, "error obtaining info");

    std::cout << "MEM=" << mem << "\n";

    // STEP 2: allocate memory
    fbs_uint_t *workspace = (fbs_uint_t *) malloc(mem*sizeof(fbs_uint_t));

    // STEP 3: init interpolator
    retVal = FBS_setup(&info, workspace);
    checked(retVal, "error initializing info");

    // STEP 4: use interpolator

    fbs_uint_t j[mm];

    // search the bin for values in z using scalar search
    for (fbs_uint_t i = 0; i < mm; ++i)
        j[i] = FBS_scalar(&zz[i], workspace);

    // show and clear results
    displayResults<L,R>(zz, mm, j, "scalar");

    // search the bin for values in z using vectorial search
    FBS_vectorial(j, zz, &mm, workspace);

    // show and clear results
    displayResults<L,R>(zz, mm, j, "vectorial");

    // STEP 5: release memory
    free(workspace);
}

template <bool L, bool R>
int demo()
{
   demo<L,R>(FBS_Direct);
   demo<L,R>(FBS_BinSearch);
   return 0;
}

int main()
{
   demo<false,false>();
   demo<false,true>();
   demo<true,true>();
   demo<true,false>();
   return 0;
}
