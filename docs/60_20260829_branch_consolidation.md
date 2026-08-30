# 2026-08-29 브랜치·worktree 통합 기록

## [가설]

현재 `dev`에는 유지할 구현이 통합돼 있으며, 과거 작업 브랜치와 별도 worktree를
제거해도 현재 개발선의 코드·검증 경계를 잃지 않을 수 있다.

## [근거]

통합 직전 read-only Git graph 감사에서 `dev`는 `main`을 포함하고 146 commits 앞서 있었다.
다음 17개 branch tip은 모두 `dev`의 ancestor였다.

```text
archive/analysis-expert-review             archive/pre-rewrite-20260827
fix/dev                                    fix/finetune-readiness-repair
work/broadband-anc-v2                      work/canonical-training
work/v10-fullband-rt5640-contract          work/v10b-rt5640-fullband-s32-admission
work/v11-full-octave-v3-admission          work/v12-synchronized-witness-admission
work/v13-rt5640-s32-capture-admission      work/v14-causal-secondary-prefix-adapter
work/v5-live-adapter-clean                 work/v6-clock-checkpoints
work/v7-nonaffine-clock                    work/v8-rt5640-zero-duplex
work/v9-electrical-frame-witness
```

`work/v5-delay-audit-hardening`은 graph상 별도였지만 current `dev`의
`4c9041a`와 stable patch-id가 같은 semantic-equivalent였다.

`work/high-frequency-validation`의 유일한 commit `d4b0c0a`는 의도적으로 merge하지
않았다. 이 commit은 60–8000 Hz USB full-duplex raw를 diagnostic으로 허용하지만,
현재 장비에서 common-clock valid repeat가 `0/64`였고 8 kHz octave 상단 11.314 kHz도
덮지 않는다. 더 중요한 것은 marker가 immutable raw가 아니라 analysis metadata에만
있어 reanalysis 승격 차단으로 불완전하다는 점이다. 현재 `dev`는 immutable band contract
검사로 그 raw를 official P/S로 승격하지 못하게 하고, `9f822e0`에서 2/4/8 kHz 단일점
진단 결과도 `results/channel_paths/` realpath 아래로 격리했다.

## [확인 방법]

각 linked worktree의 `git status --porcelain`이 비어 있는지, 해당 경로를 current working
directory로 쓰는 process가 없는지, branch containment와 `git cherry`를 확인했다.
worktree 제거는 shell 삭제가 아니라 `git worktree remove`만 사용했다.

## [결과]

다음 clean linked worktree 7개와 missing `/tmp` stale registration 하나를 Git으로
해제했다.

```text
/home/capston/Deep_ANC_v10_fullband
/home/capston/Deep_ANC_v10b_s32
/home/capston/Deep_ANC_v11_full_octave
/home/capston/Deep_ANC_v12_witness
/home/capston/Deep_ANC_v13_s32_capture
/home/capston/Deep_ANC_v14_causal_prefix
/home/capston/Deep_ANC_v5_live
/tmp/deepanc-v5-delay-hardening (registration only; directory는 이미 없었음)
```

모든 대상은 clean이었으므로 사용자 미커밋 변경을 삭제하지 않았다. raw data, model,
measurement artifact, Google Drive backup은 이 작업의 대상이 아니다.

## [판정]

**Confirmed — 개발 기준선은 `dev`, 배포 기준선은 `main`으로 단순화할 수 있다.**

`main`은 아직 canonical training/physical G4를 통과하지 않았으므로 `dev`를 병합하지
않는다. 이번 정리는 branch/worktree 이름을 없애는 일이지, 미검증 결과를 `main`에
승격하는 일이 아니다.

## [다음 행동]

로컬·origin에서 위 작업 branch를 제거한 뒤 `main`과 `dev`만 남았는지 `git worktree
list`, `git for-each-ref`, `git ls-remote --heads origin`으로 다시 확인한다. 이후 모든
새 구현은 `dev`에만 쌓고, canonical 학습·현장 검증이 끝난 뒤에만 `main`으로 병합한다.
