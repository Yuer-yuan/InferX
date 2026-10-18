"""Only control newly created children through pidfd and checked start identity."""
import json
import os
import signal
import subprocess
from pathlib import Path


def identity(pid):
    path=Path(f'/proc/{pid}')
    stat=(path/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=stat[19],uid=path.stat().st_uid,
                executable=os.readlink(path/'exe'),state=stat[0])


def same(record):
    try:
        now=identity(record['pid'])
        return all(now[k]==record[k] for k in ('pid','start_ticks','uid','executable')) and now['state']!='Z'
    except (FileNotFoundError,ProcessLookupError):
        return False


class Owned:
    def __init__(self,command,log,*,env=None,role='child'):
        if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
            raise RuntimeError('system Python with pidfd support required before creating a child')
        self.log=Path(log)
        with self.log.open('xb') as output:
            self.child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,
                                        env=env,stdin=subprocess.DEVNULL,start_new_session=True)
        self.fd=os.pidfd_open(self.child.pid)
        self.record=dict(identity(self.child.pid),command=command,host='ucloud2',log_path=str(self.log),role=role,
                         poll='verify /proc/PID/stat start_ticks and read log; owned pidfd lifecycle')
        self.closed=False

    def stop(self,timeout=20):
        try:
            if self.child.poll() is None:
                if not same(self.record):
                    raise RuntimeError('owned process identity mismatch; refusing signal')
                signal.pidfd_send_signal(self.fd,signal.SIGTERM)
                try:self.child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    if not same(self.record):raise RuntimeError('owned identity changed before kill')
                    signal.pidfd_send_signal(self.fd,signal.SIGKILL)
                    self.child.wait(timeout=10)
            return dict(reaped=self.child.poll() is not None,exit_code=self.child.returncode)
        finally:
            if not self.closed:os.close(self.fd);self.closed=True


def write_json(path,data):
    path=Path(path);tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
