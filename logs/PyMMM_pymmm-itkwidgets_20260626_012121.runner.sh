#!/usr/bin/env bash
set -o pipefail
cd /home/georgeos/GitHub/PyMMM
cmd=( bash -lc cd\ /tmp/pymmm-itkwidgets-spike\ \&\&\ export\ BROWSER=echo\ COLUMNS=300\ \&\&\ pixi\ run\ python\ -m\ itkwidgets.standalone_server\ /tmp/pymmm-itkwidgets-preview/20260331_fov0_PC_tstack_256.ngff.zarr\ --reader\ ngff_zarr\ --view-mode\ z\ --browser\ --verbose  )

exec > >(tee -a /home/georgeos/GitHub/PyMMM/logs/PyMMM_pymmm-itkwidgets_20260626_012121.log) 2>&1

echo "started: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "session: PyMMM_pymmm-itkwidgets_20260626_012121"
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
cat /home/georgeos/GitHub/PyMMM/logs/PyMMM_pymmm-itkwidgets_20260626_012121.command.txt
echo
"${cmd[@]}"
status=$?
echo
echo "finished: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "exit_status: $status"
if [[ $status -eq 0 ]]; then
  touch /home/georgeos/GitHub/PyMMM/logs/PyMMM_pymmm-itkwidgets_20260626_012121.done
else
  touch /home/georgeos/GitHub/PyMMM/logs/PyMMM_pymmm-itkwidgets_20260626_012121.failed
fi
exit $status
