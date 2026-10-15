"""Inherited Landlock boundary for local client task processes on x86_64 Linux."""
import ctypes
import os
from pathlib import Path


def restrict_agent(read_paths: list[Path], write_paths: list[Path], ports: list[int]) -> None:
    libc=ctypes.CDLL(None,use_errno=True)
    if os.uname().machine!='x86_64' or libc.syscall(444,0,0,1)<4:
        raise RuntimeError('agent tasks require x86_64 Linux Landlock ABI>=4')
    class Ruleset(ctypes.Structure):
        _fields_=[('handled_access_fs',ctypes.c_uint64),('handled_access_net',ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_=1
        _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
    class PortRule(ctypes.Structure):
        _fields_=[('allowed_access',ctypes.c_uint64),('port',ctypes.c_uint64)]
    access=(1<<15)-1
    attr=Ruleset(access,2) # CONNECT_TCP; binding remains available to the CLI itself.
    fd=libc.syscall(444,ctypes.byref(attr),ctypes.sizeof(attr),0)
    if fd<0:raise OSError(ctypes.get_errno(),'create agent ruleset')
    try:
        ro=(1<<0)|(1<<2)|(1<<3)
        for path,rights in [(p,ro) for p in read_paths]+[(p,access) for p in write_paths]:
            path=path.resolve(strict=True)
            if not path.is_dir():rights&=((1<<0)|(1<<1)|(1<<2)|(1<<14))
            parent=os.open(path,os.O_PATH|os.O_CLOEXEC)
            try:
                rule=PathRule(rights,parent)
                if libc.syscall(445,fd,1,ctypes.byref(rule),0):
                    raise OSError(ctypes.get_errno(),'agent path rule: '+str(path))
            finally:os.close(parent)
        for port in ports:
            if not 1<=port<=65535:raise ValueError('invalid TCP port')
            rule=PortRule(2,port)
            if libc.syscall(445,fd,2,ctypes.byref(rule),0):
                raise OSError(ctypes.get_errno(),'agent TCP port rule')
        if libc.prctl(38,1,0,0,0) or libc.syscall(446,fd,0):
            raise OSError(ctypes.get_errno(),'apply agent ruleset')
    finally:os.close(fd)
