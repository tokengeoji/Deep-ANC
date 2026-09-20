#!/bin/bash
# 호스트에서는 Docker만 호출하고, 프로젝트 명령은 컨테이너 안에서 실행한다.
set -euo pipefail
task_repo=$(cd "$(dirname "$0")/../.." && pwd)
task_action=${1:-help}
task_target=${2:-cpu}
task_container=deep-anc-dev
task_uid=${SUDO_UID:-$(id -u)}
task_gid=${SUDO_GID:-$(id -g)}

usage() {
  echo '사용: bash scripts/docker/dev.sh build [cpu|jetson|jetson-local]'
  echo '      bash scripts/docker/dev.sh up [cpu|jetson|jetson-local]'
  echo '      bash scripts/docker/dev.sh exec COMMAND [ARG...]'
  echo '      bash scripts/docker/dev.sh shell | status | stop | start'
}

case "$task_action" in
  build|up)
    if [[ "$task_target" != cpu && "$task_target" != jetson && "$task_target" != jetson-local ]]; then
      usage >&2
      exit 2
    fi
    if [[ "$task_target" == jetson* && "$(uname -m)" != aarch64 ]]; then
      echo 'Jetson 이미지는 ARM64 Jetson에서 빌드/실행하세요. x86에서는 cpu 검증 컨테이너를 사용합니다.' >&2
      exit 1
    fi
    if [[ "$task_target" == cpu && "$(uname -m)" != x86_64 ]]; then
      echo 'CPU 검증 이미지는 x86_64용입니다. Jetson에서는 jetson 대상을 사용하세요.' >&2
      exit 1
    fi
    task_image="deep-anc-${task_target}:dev"
    task_build_extra=()
    if [[ "$task_target" == jetson* ]]; then
      # Jetson RT 커널에서는 Docker bridge의 iptables raw 규칙 생성이 실패할 수 있다.
      # 호스트 네트워크를 사용해 시스템 방화벽/커널 설정 변경 없이 빌드한다.
      task_build_extra+=(--network host)
    fi
    if [[ "$task_action" == build ]]; then
      docker build "${task_build_extra[@]}" \
        --build-arg "DEV_UID=$task_uid" --build-arg "DEV_GID=$task_gid" \
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
    if [[ "$task_target" == jetson* ]]; then
      task_extra+=(--runtime nvidia --network host)
    fi
    if [[ "$task_target" == jetson-local ]]; then
      # 같은 R36.4 호스트의 NVIDIA 라이브러리만 연결한다. libc 등은 이미지 것을 쓴다.
      if [[ ! -r /etc/nv_tegra_release ]] || \
         [[ "$(</etc/nv_tegra_release)" != '# R36 (release), REVISION: 4.'* ]]; then
        echo 'jetson-local은 JetPack 6 / L4T R36.4 호스트가 필요합니다.' >&2
        exit 1
      fi
      task_lib_root=/usr/lib/aarch64-linux-gnu
      for task_required in libcudnn.so.9 libnvinfer.so.10 libnvonnxparser.so.10; do
        [[ -r "$task_lib_root/$task_required" ]] || {
          echo "필수 호스트 라이브러리 없음: $task_required" >&2
          exit 1
        }
      done
      task_trt=/usr/lib/python3.10/dist-packages/tensorrt
      [[ -d "$task_trt" ]] || { echo '호스트 TensorRT Python 패키지가 없습니다.' >&2; exit 1; }
      for task_lib in "$task_lib_root"/libcudnn*.so.9* \
                      "$task_lib_root"/libnvinfer*.so.10* \
                      "$task_lib_root"/libnvonnxparser*.so.10*; do
        [[ -f "$task_lib" ]] || continue
        task_extra+=(--mount "type=bind,src=$task_lib,dst=$task_lib,readonly")
      done
      task_extra+=(--mount "type=bind,src=$task_trt,dst=$task_trt,readonly")
      # 빈 venv 디렉터리의 소유권만 이미지에서 복사된다. 패키지는 up 이후 한 번 설치한다.
    fi
    # 이미지 내부 .venv를 전용 볼륨으로 복사한다. 호스트 .venv는 가려지고 사용되지 않는다.
    # 기본 컨테이너에는 사운드 장치를 노출하지 않는다.
    docker run -d --init --name "$task_container" \
      --label deep-anc.managed=true --label "deep-anc.target=$task_target" \
      --mount "type=bind,src=$task_repo,dst=/workspace/Deep-ANC" \
      --mount "type=volume,src=$task_volume,dst=/workspace/Deep-ANC/.venv" \
      --workdir /workspace/Deep-ANC "${task_extra[@]}" "$task_image"
    if [[ "$task_target" == jetson-local ]]; then
      echo '다음: bash scripts/docker/dev.sh exec bash scripts/docker/bootstrap_jetson_local.sh'
    fi
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
