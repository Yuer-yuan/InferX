"""One candidate per process. Install an irrevocable syscall filter before exec."""
import bisect
import collections
import copy
import ctypes
import decimal
import errno
import fractions
import functools
import hashlib
import heapq
import itertools
import json
import math
import os
import random
import re
import resource
import socket
import statistics
import string
import sys
import time
import typing


def restrict_syscalls():
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        # No open/openat, filesystem mutation, socket, fork/clone, exec,
        # ptrace, process_vm_*, kill, or io_uring. Only inherited pipes exist.
        for name in ("read write close fstat lseek brk mmap mprotect munmap mremap madvise "
                     "rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack futex "
                     "clock_gettime gettimeofday time getrusage getpid gettid getrandom "
                     "sched_yield restart_syscall exit exit_group").split():
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and lib.seccomp_rule_add(context, 0x7FFF0000, number, 0) != 0:
                raise RuntimeError("seccomp rule failed: " + name)
        if lib.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        lib.seccomp_release(context)


def main():
    payload = json.loads(sys.stdin.buffer.read())
    prefix = payload["marker"]
    try:
        if payload["cpu"] is not None:
            os.sched_setaffinity(0, {payload["cpu"]})
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.ceil(payload["timeout"])),) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        os.closerange(3, 1024)
        restrict_syscalls()
    except BaseException as exc:
        print(prefix + json.dumps({"status": "infrastructure_error", "error": str(exc)}), flush=True)
        return
    print(prefix + json.dumps({"status": "ready"}), flush=True)
    start = time.monotonic()
    try:
        namespace = {"__name__": "candidate"}
        exec(compile(payload["code"], "<candidate>", "exec"), namespace)
        exec(compile(payload["tests"], "<tests>", "exec"), namespace)
        namespace["check"](namespace[payload["entry_point"]])
        result = {"status": "passed", "error": None}
    except BaseException as exc:
        result = {"status": "failed", "error": type(exc).__name__ + ": " + str(exc)[:500]}
    result["test_elapsed"] = time.monotonic() - start
    print("\n" + prefix + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
