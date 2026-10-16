"""Linux repository worker boundary: Landlock ABI >=3 plus seccomp."""
import ctypes
import errno
import os
from pathlib import Path


def restrict_repository(read_paths: list[Path], scratch: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    sec = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    if os.uname().machine != 'x86_64':
        raise RuntimeError('Landlock syscall numbers only audited for x86_64')
    if libc.syscall(444, 0, 0, 1) < 3:
        raise RuntimeError('Landlock ABI >=3 required')

    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]

    attr = Ruleset((1 << 15) - 1)
    fd = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'landlock_create_ruleset')
    try:
        ro = (1 << 2) | (1 << 3)
        rw = ro | sum(1 << b for b in (1, 4, 5, 7, 8, 13, 14))
        for path, access in [(p, ro) for p in read_paths] + [(scratch, rw)]:
            path = path.resolve(strict=True)
            if not path.is_dir():
                access &= ~(1 << 3)
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access, parent)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), 'landlock_add_rule: ' + str(path))
            finally:
                os.close(parent)
        if libc.prctl(38, 1, 0, 0, 0) or libc.syscall(446, fd, 0):
            raise OSError(ctypes.get_errno(), 'landlock_restrict_self')
    finally:
        os.close(fd)
    context = sec.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise RuntimeError('seccomp_init failed')
    try:
        # Path operations are bounded by Landlock. No network, process creation,
        # exec, process inspection, signals to other processes, or io_uring.
        names = ('read write readv writev close fstat stat lstat newfstatat statx '
                 'open openat access faccessat faccessat2 lseek getdents64 readlink readlinkat '
                 'getcwd chdir fchdir fcntl ioctl dup dup2 dup3 fsync fdatasync '
                 'mkdir mkdirat unlink unlinkat rmdir rename renameat renameat2 truncate ftruncate '
                 'brk mmap mprotect munmap mremap madvise rt_sigaction rt_sigprocmask '
                 'rt_sigreturn sigaltstack futex clock_gettime clock_getres gettimeofday time '
                 'getrusage getpid getppid gettid getuid geteuid getgid getegid getgroups '
                 'getrandom uname sysinfo sched_getaffinity sched_yield restart_syscall '
                 'nanosleep clock_nanosleep poll ppoll select pselect6 exit exit_group')
        for name in names.split():
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and sec.seccomp_rule_add(context, 0x7FFF0000, number, 0):
                raise RuntimeError('seccomp rule failed: ' + name)
        if sec.seccomp_load(context):
            raise RuntimeError('seccomp_load failed')
    finally:
        sec.seccomp_release(context)
