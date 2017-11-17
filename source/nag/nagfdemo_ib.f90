! Define precision for floating point numbers
! And hence the size of the corresponding datatype
  Module precision
!   Don't use Fortran's implicit variable typing (based on first character of variable name)
    Implicit None
!   Double precision floating point numbers
    Integer, Parameter, Public       :: wp = kind(0.0D0)
    Intrinsic                        :: kind  
  End Module precision


! Define integer values corresponding to C enum values in nagfbs.h
! and derived types corresponding to C structs in nagfbs.h
  Module fbstypes

!   We need to access Fortran's C bindings
    Use iso_c_binding

!   Use our definition of floating point number precision
    Use precision, Only: wp

!   Don't use Fortran's implicit variable typing (based on first character of variable name)
    Implicit None

!   enums and return types
!   Parameters are analogous to C/C++ consts
    Integer, Parameter :: FBS_Direct            = 0
    Integer, Parameter :: FBS_BinSearch         = 1

    Integer, Parameter :: FBS_CODE_UNKNOWN      = 1
    Integer, Parameter :: FBS_DIRECT_UNFEASIBLE = 2
    Integer, Parameter :: FBS_X_TOO_LARGE       = 3
    Integer, Parameter :: FBS_NOT_IMPLEMENTED   = 100

!   Fortran derived type corresponding to DirectInfo struct
    Type, Bind(c) :: directinfo
      Real(kind=wp) :: scaler
    End Type directinfo

!   Fortran derived type corresping to FBSInfo struct
    Type, Bind(c) :: fbsinfo
      Type(c_ptr) :: x
      Integer :: n
      Integer :: code
      Logical(c_bool) :: outL, outR
      Type(directinfo) :: direct
    End Type fbsinfo

  End Module fbstypes


! Define the Fortran interfaces for C FBS functions
  Module nagfdemo_ib
!   Don't use Fortran's implicit variable typing (based on first character of variable name)  
    Implicit None
    Interface
!     Fortran interface for FBS_getInfo
      Integer Function FBS_getInfo(code, x, n, outLeft, outRight, mem, info)
        Use iso_c_binding, Only: c_bool
        Use precision, Only: wp
        Use fbstypes, Only: fbsinfo
        Implicit None
        Integer, Intent (In) :: code
        Integer, Intent (In) :: n
        Real(kind=wp), Intent (In) :: x(*)
        Logical(c_bool), Intent (In) :: outLeft, outRight
        Integer, Intent (Out) :: mem
        Type(fbsinfo), Intent (Out) :: info
      End Function FBS_getInfo
!     Fortran interface for FBS_setup
      Integer Function FBS_setup(info, cwork)
        Use fbstypes, Only: fbsinfo
        Implicit None
        Type(fbsinfo), Intent (In) :: info
        Integer, Intent (Out) :: cwork(*)
      End Function FBS_setup
!     Fortran interface for FBS_scalar
      Integer Function FBS_scalar(z, cwork)
        Use precision, Only: wp
        Implicit None
        Real (Kind=wp), Intent (In) :: z
        Integer, Intent (In) :: cwork(*)
      End Function FBS_scalar
!     Fortran interface for FBS_vectorial
      Integer Function FBS_vectorial(j, z, m, cwork)
        Use precision, Only: wp
        Implicit None
        Integer, Intent (Out) :: j(*)
        Real (Kind=wp), Intent (In) :: z(*)
        Integer, Intent (In) :: m
        Integer, Intent (In) :: cwork(*)
      End Function FBS_vectorial
    End Interface
  End Module nagfdemo_ib
  
    
