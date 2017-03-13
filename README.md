# FastBinarySearch
Source code demonstrating and testing fast vectorizable algorithms for searching for the insertion point in a sorted vector of floating point numbers. The associated paper is available at https://arxiv.org/abs/1506.08620. The abstract section is copied here below.

# Fast and Vectorizable Alternative to Binary Search in O(1) Applicable to a Wide Domain of Sorted Arrays of Floating Point Numbers

Given an array X of N+1 strictly ordered floating point numbers and a set of M floating point numbers Z belonging to the interval [X[0],X[N]), a common problem in numerical methods algorithms is to find the indices of the largest numbers in the array X which are smaller or equal than the numbers Z[j].

This problem arises for instance in the context of piece-wise interpolation, where a domain [X[0],X[N]) is partitioned in sub-intervals {[X[i],X[i+1]} and different interpolation functions gi(x) are associated with each sub-interval. To compute the interpolated value for a number Z[j], the index i of the sub-interval containing it needs to be resolved first.

The general solution to this problem is the binary search algorithm, which has complexity O(M log2 N). The classical and well known implementation of the algorithm requires a control flow branch, which incurs penalties on many CPU architectures, and is not vectorizable, i.e. it does not benefit from the vectorial capabilities of modern CPUs.

In some special cases, when either the X[i] or the Z[j] numbers exhibit particular patterns, more efficient algorithms are available. Examples are when the numbers X[i] are equally spaced or when the numbers Z[j] are sorted, where the problem can be solved with complexity O(M) and O(M+N) respectively. However no generic alternative exists.

This paper describes improvements to the binary search algorithm, which make it faster and vectorizable.

Next it proposes a new vectorizable algorithm applicable to a wide set of X partitions, which is based on an indexing technique and allows to solve the problem with complexity O(1) per individual search at the cost of introducing an initial overhead to compute the index and requiring extra memory for its storage.

Test results using streaming SIMD extensions compare the performance of the algorithm versus various benchmarks and demonstrate its effectiveness. Depending on the test case, the algorithm performs up to 43 times faster than the classical binary search in single precision and 39 times faster in double precision.

Applicability limitations and cache-friendliness related aspects are also discussed.
