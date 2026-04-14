#ifndef PLATFORM_H
#define PLATFORM_H

/* ================================================================
 * Cross-platform threading and timing
 * ================================================================ */

#ifdef _WIN32
  #include <windows.h>

  typedef HANDLE thread_t;
  typedef DWORD (WINAPI *thread_fn_t)(LPVOID);

  static inline int thread_create(thread_t *t, void *(*fn)(void*), void *arg) {
      *t = CreateThread(NULL, 0, (thread_fn_t)fn, arg, 0, NULL);
      return *t ? 0 : -1;
  }
  static inline void thread_join(thread_t t) {
      WaitForSingleObject(t, INFINITE);
      CloseHandle(t);
  }

  static inline double platform_now_sec(void) {
      LARGE_INTEGER freq, cnt;
      QueryPerformanceFrequency(&freq);
      QueryPerformanceCounter(&cnt);
      return (double)cnt.QuadPart / (double)freq.QuadPart;
  }

#else
  #include <pthread.h>
  #include <time.h>

  typedef pthread_t thread_t;

  static inline int thread_create(thread_t *t, void *(*fn)(void*), void *arg) {
      return pthread_create(t, NULL, fn, arg);
  }
  static inline void thread_join(thread_t t) {
      pthread_join(t, NULL);
  }

  static inline double platform_now_sec(void) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
  }

#endif

#endif /* PLATFORM_H */
