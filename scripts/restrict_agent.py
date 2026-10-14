"""Apply task-only restrictions after client environment setup, then exec the CLI."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferx.agent_restrictions import restrict_agent

root=Path(os.environ['INFERX_ROOT']).resolve()
workspace=Path.cwd().resolve()
if not workspace.is_relative_to(root/'context-agents'):
    raise RuntimeError('restricted task workspace must be within the HDD experiment directory')
read=[Path(x) for x in ['/usr','/bin','/lib','/lib64','/etc','/proc','/sys','/dev/urandom'] if Path(x).exists()]
read += [root/'toolchains/node-v22.23.3-linux-x64',root/'repo-venv',root/'agent-clients',
         root/'context-agents/config',Path('/home/guokc/.local/bin/claude').resolve()]
write=[workspace,root/'agent-state',root/'cache',root/'tmp',Path('/dev/null')]
restrict_agent(read,write,[18081,18082])
command=sys.argv[1:]
if command and command[0]=='--':command=command[1:]
if not command:raise RuntimeError('missing client command')
os.execvpe(command[0],command,os.environ)
