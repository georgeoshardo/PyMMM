#!/usr/bin/env bash
set -o pipefail
cd /home/georgeos/GitHub/PyMMM
cmd=( bash -lc cd\ /home/georgeos/GitHub/PyMMM\ \&\&\ export\ PYMMM_ITKWIDGETS_VIEWER_URL=\"\$\(grep\ -o\ \"http://127.0.0.1:37480/itkwidgets/index.html\?workspace=.\*\"\ logs/PyMMM_pymmm-itkwidgets_20260626_012121.log\ \|\ tail\ -n\ 1\)\"\ \&\&\ pixi\ run\ marimo\ edit\ notebooks/pymmm_itkwidgets_browser.py\ --port\ 2720\ --no-token  )

exec > >(tee -a /home/georgeos/GitHub/PyMMM/logs/PyMMM_pymmm-itkwidgets-marimo_20260626_012143.log) 2>&1

echo "started: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "session: PyMMM_pymmm-itkwidgets-marimo_20260626_012143"
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
cat /home/georgeos/GitHub/PyMMM/logs/PyMMM_pymmm-itkwidgets-marimo_20260626_012143.command.txt
echo
"${cmd[@]}"
status=$?
echo
echo "finished: $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "exit_status: $status"
if [[ $status -eq 0 ]]; then
  touch /home/georgeos/GitHub/PyMMM/logs/PyMMM_pymmm-itkwidgets-marimo_20260626_012143.done
else
  touch /home/georgeos/GitHub/PyMMM/logs/PyMMM_pymmm-itkwidgets-marimo_20260626_012143.failed
fi
exit $status
