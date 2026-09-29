#!/usr/bin/env bash
# Every host-side tool runs in the container, so the VS Code entries call this
# instead of each carrying its own one-line docker incantation. Add a target
# here rather than a longer string in launch.json.
set -euo pipefail
cd "$(dirname "$0")/.."

gui() {   # gui <container name> <service> <command...>
  local name=$1 service=$2
  shift 2
  xhost +local: >/dev/null 2>&1 || true        # the container's Qt onto this X server
  docker rm -f "$name" >/dev/null 2>&1 || true # clear one left behind by a hard kill
  exec docker compose run --rm \
    -e QT_X11_NO_MITSHM=1 \
    -e "DISPLAY=${DISPLAY:-:0}" \
    --name "$name" \
    "$service" "$@"
}

run() {   # run <service> <command...>
  local service=$1
  shift
  exec docker compose run --rm "$service" "$@"
}

case "${1:-}" in
  viewer)                     # waterfalls, camera, depth, REC, protocols
    gui csi_live csi \
      bash -lc 'cd /workspace/tools && exec python3 viewer.py' ;;

  camera)                     # camera + depth only, for placing the camera
    gui csi_cam csi \
      bash -lc 'cd /workspace/tools && exec python3 camera_view.py' ;;

  pose)                       # live skeleton + top-down, REC a segment, fit, 3-D result
    gui pose_live pose \
      python3 tools/live_body.py "${@:2}" ;;

  record)                     # record <seconds> <prefix>; stop the GUI first
    run csi \
      bash -lc "cd /workspace/tools && python3 capture.py --seconds ${2:-60} --prefix ${3:-test/take}" ;;

  check)                      # check <session>
    run csi \
      bash -lc "cd /workspace && python3 tools/check_session.py data/${2:?session}" ;;

  finish)                     # finish <session>: add the CSI windows to raw takes
    run csi \
      bash -lc "cd /workspace && python3 tools/finish_capture.py data/${2:?session}" ;;

  firmware)
    run csi \
      bash -lc 'cd /workspace/firmware && idf.py build' ;;

  *)
    echo "usage: ${0##*/} {viewer|camera|pose|record|check|finish|firmware} [args]" >&2
    exit 2 ;;
esac
