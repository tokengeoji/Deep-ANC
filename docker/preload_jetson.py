"""컨테이너 venv에만 기존 Jetson 필수 라이브러리 preload 훅을 설치한다."""

import os
from pathlib import Path
import sys
import sysconfig


def main() -> None:
    # BuildKit의 RUN 샌드박스에는 일반 `docker run`과 달리 /.dockerenv가 없다.
    # Dockerfile의 환경 표식과 venv를 함께 확인해 Jetson 컨테이너 설치 계약을 강제한다.
    if sys.prefix == sys.base_prefix or os.environ.get('DEEP_ANC_CONTAINER') != 'jetson':
        raise SystemExit('Docker 컨테이너의 venv에서만 설치할 수 있습니다')
    site = Path(sysconfig.get_path('purelib'))
    hook = '''import ctypes
import os
from pathlib import Path
nv = Path(__file__).resolve().parent / "nvidia"
for relative in ("nvtx/lib/libnvToolsExt.so.1", "cuda_cupti/lib/libcupti.so.12",
                 "cusparselt/lib/libcusparseLt.so.0"):
    library = nv / relative
    if library.exists():
        ctypes.CDLL(str(library), mode=os.RTLD_GLOBAL)
'''
    (site / '_deep_anc_libpaths.py').write_text(hook, encoding='utf-8')
    (site / '_deep_anc_libs.pth').write_text('import _deep_anc_libpaths\n', encoding='utf-8')


if __name__ == '__main__':
    main()
