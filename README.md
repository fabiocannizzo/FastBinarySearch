# FastBinarySearch
Source code demonstrating fast vectorizable algorithms for searching for the insertion point in a sorted vector of floating point numbers.

Given an array X of N+1 strictly ordered floating point numbers and a set of M floating point numbers Z belonging to the interval [X[0],X[N]), a common problem in numerical methods algorithms is to find the indices of the largest numbers in the array X which are smaller or equal than the numbers Z[j].

This problem arises for instance in the context of piece-wise interpolation, where a domain [X[0],X[N]) is partitioned in sub-intervals {[X[i],X[i+1]} and different interpolation 
functions gi(x) are associated with each sub-interval. To compute the interpolated value for a number Z[j], the index i of the sub-interval containing it needs to be resolved first.

The general solution to this problem is the binary search algorithm, which has complexity O(M log2 N). The classical and well known implementation of the algorithm requires a control flow branch, which incurs penalties on many CPU architectures, and is not vectorizable, i.e. it does not benefit from the vectorial capabilities of modern CPUs.

In some special cases, when either the X[i] or the Z[j] numbers exhibit particular patterns, more efficient algorithms are available. Examples are when the numbers X[i] are equally spaced or when the numbers Z[j] are sorted, where the problem can be solved with complexity O(M) and O(M+N) respectively. However no generic alternative exists.

This paper describes an improvement to the binary search algorithm, which avoid the control flow branch, thus making it generally faster. The complexity of the algorithm is still O(M log_2 N), but performance improves by a proportionality factor a/d, where a is a constant smaller than one associated with the performance gain due to the removal of the branch and, d is the number of floating point numbers which can be processed simultaneously (assuming perfect vectorization) and depends on the chosen set of vectorial instructions and floating point representation (e.g. with SSE instructions in single precision d=4). 

Next it proposes a new vectorizable algorithm based on a indexing technique, which reduces complexity of search operations to O(M), at the cost of introducing an initial overhead to compute the index and requiring extra memory for its storage. The algorithm has general applicability, but the relative magnitude of such extra costs, which are related to the layout of the numbers X[i], in some particular cases might make its use not efficient.

Some benchmark test results reproducible with the source code in this repository demonstrate that the proposed algorithm is about 26 times faster than the classical binary search in single precision and 15 times faster in double precision (test results are dependent on the hardware used).