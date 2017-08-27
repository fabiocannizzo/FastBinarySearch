// dllmain.cpp : Defines the entry point for the DLL application.
// This file is used in cygwin to wrap the MKL DLL

#include <SDKDDKVer.h>
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#include <windows.h>

#include "mklloader.h"

#define MYEXPORT(RetType, Name, Arguments) extern "C" RetType __declspec(dllexport) Name##_wrap Arguments

MYEXPORT(int, mysum, (int x, int y))
{
    return x + y;
}

MYEXPORT(int, dfdNewTask1D, (DFTaskPtr *task, const MKL_INT nx, const double x[], const MKL_INT xhint, const MKL_INT ny, const double y[], const MKL_INT yhint))
{
    return dfdNewTask1D(task, nx, x, xhint, ny, y, yhint);
}

MYEXPORT(int, dfsNewTask1D, (DFTaskPtr *task, const MKL_INT nx, const float x[], const MKL_INT xhint, const MKL_INT ny, const float y[], const MKL_INT yhint))
{
    return dfsNewTask1D(task, nx, x, xhint, ny, y, yhint);
}

MYEXPORT(int, dfdSearchCells1D, (DFTaskPtr task, const MKL_INT method, const MKL_INT n, const double x[], const MKL_INT hint, const double dh[], MKL_INT res[]))
{
    return dfdSearchCells1D(task, method, n, x, hint, dh, res);
}

MYEXPORT(int, dfsSearchCells1D, (DFTaskPtr task, const MKL_INT method, const MKL_INT n, const float x[], const MKL_INT hint, const float dh[], MKL_INT res[]))
{
    return dfsSearchCells1D(task, method, n, x, hint, dh, res);
}

MYEXPORT(int, dfDeleteTask, (DFTaskPtr *task))
{
    return dfDeleteTask(task);
}

#define TEST_HEADER(name) name##_t p_##name = & name##_wrap

void test_signature()
{
    TEST_HEADER(dfdNewTask1D);
    TEST_HEADER(dfsNewTask1D);
    TEST_HEADER(dfdSearchCells1D);
    TEST_HEADER(dfsSearchCells1D);
    TEST_HEADER(dfDeleteTask);
}

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
					 )
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
        test_signature();
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}

