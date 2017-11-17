  Program nagfdemo
!   Ensure C bools and Fortran logicals can interoperate
    Use iso_c_binding, Only: c_bool

!   Import our definition of how big a floating point number is
    Use precision, Only: wp

!   Import our user-defined FBS types and enumerations
    Use fbstypes

!   Import the function interface for FBS_getInfo    
    Use nagfdemo_ib, Only: FBS_getInfo, FBS_setup, FBS_scalar, FBS_vectorial

!   Don't use Fortran's implicit variable typing (based on first character of variable name)
    Implicit None

!   Sample data
!   Parameters are analogous to C consts
    Real(kind=wp), Parameter :: x(6) =  (/ 1.0, 1.5, 2.0, 4.0, 5.0, 9.0 /)
    Real(kind=wp), Parameter :: z(8) =  (/ 1.0, 2.0, 4.0, 5.0, 1.5, 2.5, 4.8, 8.2 /)

!   Variables to use in the program
    Integer :: retVal
    Integer :: i, mem, m, n
    Logical(c_bool) :: oL, oR
    Type(fbsinfo) :: info

    Integer, Allocatable :: cwork(:)
    Integer, Allocatable :: j(:)
    
!   Import the size() intrinsic, to allow us to query array sizes    
    Intrinsic :: size

!   Initial values for all the variables
!   Set some to -1 so we can check that they change
    retval = -1
    m = size(z)
    n = size(x)
    mem = -1
    oL = .false.
    oR = .false.

!   write(*,*) "x = ", x
!   write(*,*) "n = ", n
!   write(*,*) "mem = ", mem

    ! // STEP 1: get info
    ! retVal = FBS_getInfo(code, x, n, false, false, mem, &info);
    ! checked(retVal, "error obtaining info");
    ! std::cout << "MEM=" << mem << "\n";
    ! MEM=497 for direct search
    
!   Call FBS_getInfo and write out some returned information    
    retval = FBS_getInfo(FBS_Direct, x, n, oL, oR, mem, info)
!    retval = FBS_getInfo(FBS_BinSearch, x, n, oL, oR, mem, info)
    write(*,*) "FBS_getInfo() retval = ", retval
    write(*,*) "mem = ", mem
    
    ! // STEP 2: allocate memory
    ! char *workspace = (char *) malloc(mem);
    
    Allocate(cwork(mem))

    ! // STEP 3: init interpolator
    ! retVal = FBS_setup(&info, workspace);
    ! checked(retVal, "error initializing info");

!   Call FBS_setup and write out some returned information        
    retval = FBS_setup(info, cwork)
    write(*,*) "FBS_setup() retval = ", retval
    
    ! // STEP 4: use interpolator

    ! fbs_uint_t j[m];
    Allocate(j(m))
    
    ! // search the bin for values in z using scalar search
    ! for (size_t i = 0; i < m; ++i)
    !     j[i] = FBS_scalar(z[i], workspace);  
    ! // show and clear results
    ! displayResults(j, "scalar");

    !  scalar results:
    ! 1 <= 1 < 1.5
    ! 2 <= 2 < 4
    ! 4 <= 4 < 5
    ! 5 <= 5 < 9
    ! 1.5 <= 1.5 < 2
    ! 2 <= 2.5 < 4
    ! 4 <= 4.8 < 5
    ! 5 <= 8.2 < 9
    
    do i=1,m
      j(i) = FBS_scalar(z(i), cwork)
    end do

!   We would expect:
!      1 3 4 5 2 3 4 5
    write(*,*) "Scalar results:"
    write(*,*) j

    ! // search the bin for values in z using vectorial search
    ! FBS_vectorial(j, z, m, workspace);
    ! // show and clear results
    ! displayResults(j, "vectorial");

    !    vectorial results:
    ! 1 <= 1 < 1.5
    ! 2 <= 2 < 4
    ! 4 <= 4 < 5
    ! 5 <= 5 < 9
    ! 1.5 <= 1.5 < 2
    ! 2 <= 2.5 < 4
    ! 4 <= 4.8 < 5
    ! 5 <= 8.2 < 9

!   Zero the return array for second call
    j(1:m) = 0
    
    retval = FBS_vectorial(j, z, m, cwork)
    write(*,*) "FBS_vectorial() retval = ", retval

!   We would expect:
!      1 3 4 5 2 3 4 5
    write(*,*) "Vectorial results:"
    write(*,*) j
    
    ! // STEP 5: release memory
    ! free(workspace);

    Deallocate(cwork)
    Deallocate(j)
    
  End Program nagfdemo
  
