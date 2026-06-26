#!/usr/bin/env bash
set -o pipefail
cd /home/georgeos/GitHub/PyMMM
cmd=( pixi run python scripts/pymmm_nd2_web_viewer.py --nd2 /data/20260331_SB5_6_7_8_16/20260331.nd2 --port 2730  )

exec > >(tee -a /home/georgeos/GitHub/PyMMM/logs/PyMMM_pymmm-nd2-web-viewer_20260626_013001.log) 2>&1

echo "started: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "session: PyMMM_pymmm-nd2-web-viewer_20260626_013001"
echo "cwd: $(pwd)"
echo
echo "git:"
git status --short --branch 2>/dev/null || true
git rev-parse HEAD 2>/dev/null || true
echo
echo "environment:"
if command -v pixi >/dev/null 2>&1; then
  pixi --version || true
  if [[ -f pixi.toml ]] || [[ -f pyproject.toml ]]; then
    pixi run python --version || true
  fi
fi
echo
echo "command:"
cat /home/georgeos/GitHub/PyMMM/logs/PyMMM_pymmm-nd2-web-viewer_20260626_013001.command.txt
echo
"${cmd[@]}"
status=$?
echo
echo "finished: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "exit_status: $status"
if [[ $status -eq 0 ]]; then
  touch /home/georgeos/GitHub/PyMMM/logs/PyMMM_pymmm-nd2-web-viewer_20260626_013001.done
else
  touch /home/georgeos/GitHub/PyMMM/logs/PyMMM_pymmm-nd2-web-viewer_20260626_013001.failed
fi
exit $status
