import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path


def test_client_boundary_allows_workspace_and_local_api_but_denies_other_writes_and_ports(tmp_path):
    from inferx.agent_restrictions import restrict_agent
    workspace=tmp_path/'workspace';workspace.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    allowed=socket.socket();allowed.bind(('127.0.0.1',0));allowed.listen()
    denied=socket.socket();denied.bind(('127.0.0.1',0));denied.listen()
    program='''
import json,socket,sys
from pathlib import Path
from inferx.agent_restrictions import restrict_agent
workspace=Path(sys.argv[1]);outside=Path(sys.argv[2])
restrict_agent([Path('/usr'),Path('/lib'),Path('/bin')],[workspace],[int(sys.argv[3])])
(workspace/'ok').write_text('ok')
try:(outside/'bad').write_text('bad')
except PermissionError:pass
else:raise AssertionError('outside write allowed')
s=socket.socket();s.settimeout(1);s.connect(('127.0.0.1',int(sys.argv[3])));s.close()
s=socket.socket();s.settimeout(1)
try:s.connect(('127.0.0.1',int(sys.argv[4])))
except PermissionError:pass
else:raise AssertionError('disallowed TCP port allowed')
print('bounded')
'''
    try:
        p=subprocess.run([sys.executable,'-c',program,str(workspace),str(outside),
            str(allowed.getsockname()[1]),str(denied.getsockname()[1])],capture_output=True,text=True,timeout=10)
        assert p.returncode==0,p.stderr
        assert p.stdout.strip()=='bounded'
        assert (workspace/'ok').read_text()=='ok' and not (outside/'bad').exists()
    finally:allowed.close();denied.close()
