"""컨테이너 venv에만 기존 Jetson 필수 라이브러리 preload 훅을 설치한다."""

from pathlib import Path
import sys
import sysconfig


def main() -> None:
    if sys.prefix == sys.base_prefix or not Path('/.dockerenv').exists():
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
