#pragma once

// ------------------------------------------------------------------
// OS abstraction for threading, sockets, memory mapping
// ------------------------------------------------------------------

#ifdef _WIN32
    #include <winsock2.h>
    #include <windows.h>
    #include <process.h>
    typedef HANDLE thread_t;
    #define thread_return_t unsigned __stdcall
    #define thread_func(name) unsigned __stdcall name(void* arg)
#else
    #include <pthread.h>
    #include <unistd.h>
    #include <sys/mman.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <fcntl.h>
    typedef pthread_t thread_t;
    #define thread_return_t void*
    #define thread_func(name) void* name(void* arg)
#endif

#include <cstdint>
#include <cstring>

inline void thread_create(thread_t* t, thread_return_t (*func)(void*), void* arg) {
#ifdef _WIN32
    *t = (HANDLE)_beginthreadex(NULL, 0, func, arg, 0, NULL);
#else
    pthread_create(t, NULL, func, arg);
#endif
}

inline void thread_join(thread_t t) {
#ifdef _WIN32
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
#else
    pthread_join(t, NULL);
#endif
}

inline void thread_yield() {
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
}
