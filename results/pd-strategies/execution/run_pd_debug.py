"""Diagnostic only: same controller/restoration, shorter native watchdog."""
from scripts import run_pd_baselines as controller
original=controller.OwnedEngine.start
def start(server):
 server.args.extend(server.case.get('diagnostic_extra_flags',[]))
 return original(server)
controller.OwnedEngine.start=start
controller.main()
