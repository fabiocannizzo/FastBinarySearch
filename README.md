# FastBinarySearch
This source code was originally written to support the conclusions of an article recently published on the Journal of parallel and Distributed Computing. It demonstrates and test fast vectorizable algorithms for searching for the insertion point in a sorted vector of floating point numbers. The article is available at https://authors.elsevier.com/a/1W50B2f6jyvQWq and a preprint draft is available at https://arxiv.org/abs/1506.08620. The abstract section is copied here below.

Since then the code has been refactored to be usable as a header-only library. It is very easy to use the library, it only takes a few lines of code. Just include BinSearch.h, create a search engine, profit. A demo program is provided in the source/demo subdirectory.

A C and a Fortran simple API with external memory management are also available. They only allow to use a small subset of the features in the library and have been designed specifically to support future inclusion of these features in the NAG library.

Some performance test results are available below.

If you are interested in using the library for any purpose and need some help, let me know.

# Fast and Vectorizable Alternative to Binary Search in O(1) Applicable to a Wide Domain of Sorted Arrays of Floating Point Numbers

Given an array X of N+1 strictly ordered floating point numbers and a floating point number z belonging to the interval [X[0],X[N]), a common problem in numerical methods is to find the index i of the interval [X[i],X[i+1]) containing z, i.e. the index of the largest number in the array X which is smaller or equal than z. This problem arises for instance in the context of spline interpolation or the computation of empirical probability distribution from empirical data. Often it needs to be solved for a large number of different values z and the same array X, which makes it worth investing resources upfront in pre-processing the array X with the goal of speeding up subsequent search operations. In some cases the values z to be processed are known simultaneously in blocks of size M, which offers the opportunity to solve the problem vectorially, exploiting the parallel capabilities of modern CPUs. The common solution is to sequentially invoke M times the well known binary search algorithm, which has complexity O(log2N) per individual search and, in its classic formulation, is not vectorizable, i.e. it is not SIMD friendly. This paper describes technical improvements to the binary search algorithm, which make it faster and vectorizable. Next it proposes a new vectorizable algorithm, based on an indexing technique, applicable to a wide family of X partitions, which solves the problem with complexity O(1) per individual search at the cost of introducing an initial overhead to compute the index and requiring extra memory for its storage. Test results using streaming SIMD extensions compare the performance of the algorithm versus various benchmarks and demonstrate its effectiveness. Depending on the test case, the algorithm can produce a throughput up to two orders of magnitude larger than the classic binary search. Applicability limitations and cache-friendliness related aspects are also discussed.

# Some performance test results
The numbers shown in the table below means throughput, i.e. the larger the better. The throughput is expressed in millions of searches per second. Various algorithms are benchmarked for searching in an array of size 2048 in single or double precision. Results are provided for searches in scalar mode (search one number at a time) of vectorial mode (search for an array of numbers as a bulk query). In vectorial mode results are provided for SSE-2 and AVX-2 instruction set. In some cases FMA are also used.

For reference, the algorithm named LowerBound is the function lower_bound as implemented in the STL distributed with gcc 7.

The test results below are obtained on a Intel Xeon CPU, model E5-2620 v3 @ 2.40GHz.

                    |          Single              |          Double
                    |  Scalar     SSE-4     AVX-2  |  Scalar     SSE-4     AVX-2
                   -------------------------------------------------------------
    DirectCacheFMA  |  496.44    480.42    930.82  |  496.44    480.42    465.41
         DirectFMA  |  465.41    480.42    992.87  |  496.44    480.42    930.82
        Direct2FMA  |  232.70    992.87    480.42  |  240.21    316.87    930.82
       DirectCache  |  465.41    480.42    930.82  |  240.21    480.42    480.42
            Direct  |  496.44    465.41    992.87  |  465.41    480.42    930.82
           Direct2  |  240.21    465.41    480.42  |  240.21    316.87    480.42
            Nonary  |  120.11       ---       ---  |     ---       ---       ---
           Pentary  |   68.32       ---       ---  |   68.32       ---       ---
           Ternary  |     ---       ---       ---  |   53.19       ---       ---
         Eytzinger  |   39.61     63.65     87.09  |   36.68     43.42     53.00
            BitSet  |   47.73     68.00    106.38  |   47.73     53.00     53.00
     ClassicOffset  |   39.82     68.00     87.09  |   43.55     43.29     50.31
       MorinOffset  |   39.82       ---       ---  |   39.82       ---       ---
       BitSetNoPad  |   47.73     73.36     95.47  |   47.73     47.73     53.00
        ClassicMod  |   28.10     50.14     73.36  |   29.79     31.82     39.82
      MorinBranchy  |   14.49       ---       ---  |   14.92       ---       ---
           Classic  |   14.46       ---       ---  |   14.02       ---       ---
        LowerBound  |   14.02       ---       ---  |   13.64       ---       ---
