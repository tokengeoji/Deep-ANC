#!/bin/bash
# 호스트에서는 Docker만 호출하고, 프로젝트 명령은 컨테이너 안에서 실행한다.
set -euo pipefail
task_repo=$(cd "$(dirname "$0")/../.." && pwd)
task_action=${1:-help}
task_target=${2:-cpu}
task_container=deep-anc-dev

usage() {
  echo '사용: bash scripts/docker/dev.sh build [cpu|jetson]'
  echo '      bash scripts/docker/dev.sh up [cpu|jetson]'
  echo '      bash scripts/docker/dev.sh exec COMMAND [ARG...]'
  echo '      bash scripts/docker/dev.sh shell | status | stop | start'
}

case "$task_action" in
  build|up)
    if [[ "$task_target" != cpu && "$task_target" != jetson ]]; then
      usage >&2
      exit 2
    fi
    if [[ "$task_target" == jetson && "$(uname -m)" != aarch64 ]]; then
      echo 'Jetson 이미지는 ARM64 Jetson에서 빌드/실행하세요. x86에서는 cpu 검증 컨테이너를 사용합니다.' >&2
      exit 1
    fi
    if [[ "$task_target" == cpu && "$(uname -m)" != x86_64 ]]; then
      echo 'CPU 검증 이미지는 x86_64용입니다. Jetson에서는 jetson 대상을 사용하세요.' >&2
      exit 1
    fi
    task_image="deep-anc-${task_target}:dev"
    if [[ "$task_action" == build ]]; then
      docker build --build-arg "DEV_UID=$(id -u)" --build-arg "DEV_GID=$(id -g)" \
        -f "$task_repo/docker/Dockerfile.$task_target" -t "$task_image" "$task_repo"
      exit
    fi
    if docker container inspect "$task_container" >/dev/null 2>&1; then
      echo 'deep-anc-dev 컨테이너가 이미 있습니다. status/exec를 사용하세요. 기존 환경은 자동 교체하지 않습니다.' >&2
      exit 1
    fi
    task_image_id=$(docker image inspect --format '{{.Id}}' "$task_image")
    task_image_id=${task_image_id#sha256:}
    task_volume="deep-anc-${task_target}-venv-${task_image_id:0:12}"
    task_extra=()
    # 저장소가 부모 Git worktree 아래에 있는 현재 체크아웃도 git diff/status로 읽는다.
    if [[ ! -e "$task_repo/.git" && -d "$task_repo/../.git" ]]; then
      task_extra+=(--mount "type=bind,src=$task_repo/../.git,dst=/workspace/.git,readonly")
    fi
    if [[ "$task_target" == jetson ]]; then
      task_extra+=(--runtime nvidia)
    fi
    # 이미지 내부 .venv를 전용 볼륨으로 복사한다. 호스트 .venv는 가려지고 사용되지 않는다.
    # 기본 컨테이너에는 사운드 장치를 노출하지 않는다.
    docker run -d --init --name "$task_container" \
      --label deep-anc.managed=true --label "deep-anc.target=$task_target" \
      --mount "type=bind,src=$task_repo,dst=/workspace/Deep-ANC" \
      --mount "type=volume,src=$task_volume,dst=/workspace/Deep-ANC/.venv" \
      --workdir /workspace/Deep-ANC "${task_extra[@]}" "$task_image"
    ;;
  exec)
    shift
    [[ $# -gt 0 ]] || { usage >&2; exit 2; }
    docker exec -i "$task_container" "$@"
    ;;
  shell)
    docker exec -it "$task_container" bash
    ;;
  status)
    docker container inspect --format '{{.Name}} {{.State.Status}} {{.Config.Image}}' "$task_container"
    ;;
  stop)
    docker stop "$task_container"
    ;;
  start)
    [[ "$(docker container inspect --format '{{index .Config.Labels "deep-anc.managed"}}' "$task_container")" == true ]] || {
      echo '이 스크립트가 만든 컨테이너가 아닙니다. 시작하지 않습니다.' >&2
      exit 1
    }
    docker start "$task_container"
    ;;
  *) usage; [[ "$task_action" == help ]] ;;
esac
