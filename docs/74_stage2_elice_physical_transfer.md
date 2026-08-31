# Stage-2 2 kHz Elice 물리 P/S 전송 계약

> [!IMPORTANT]
> 이 문서의 `combined-generation(101세션)` 표기는 Stage-1 historical transfer shape를
> 설명하기 위한 것이다. 2026-09-01 현행 Stage-2 authority는 47개 독립 component
> (`stage2-2khz-47slot-v1`, `train/val/test=16/15/16`)와 same-card RT5640/J511 actual
> P/S raw/plant binding을 모두 요구한다. USB AB13X output-master 경로는 split-clock
> failure로 retired forensic-only이며, 이 문서가 bootstrap 또는 학습을 여는 명령이 아니다.
> 현재 gate는 `HANDOFF.md` 최상단과 `docs/75`를 우선한다.

Stage-2 scratch pretrain은 일반 recorded 전송 manifest(v1)나 combined recorded
generation 전송 manifest(v2)만으로 시작할 수 없다. 실제 덕트에서 strict P/S가
PASS하고, 사람이 검토한 Git authority가 exact commit에 포함된 뒤에만
**transfer schema v3**를 발행한다.

## v3가 추가로 보내는 typed role

| Role | 고정 조건 | 검증 |
| --- | --- | --- |
| `stage2_2khz_plant_binding` | `results/.../plant_binding.json` 1개 | manifest path/SHA, physical loader의 raw/P/S/level/clock 재해시 |
| `stage2_2khz_physical_authority` | `authority/stage2_2khz_physical.json` 정확히 1개 | manifest path/SHA와 tracked exact-HEAD authority 재해시 |

상위 `stage2_2khz_physical` pointer는 두 role의 상대경로와 SHA-256을 중복해
가리킨다. 어느 하나라도 없거나 서로 다르면 manifest 생성, 로컬 전송 전 검증,
Elice bootstrap, Stage-2 public-data issuer 순서 중 가장 이른 지점에서 중단한다.
v1/v2는 일반 recorded workflow에는 남아 있지만 Stage-2 canonical admission에는
사용할 수 없다.

## 발행 순서

1. strict Stage-2 P/S raw/analysis/NPZ와 `plant_binding.json`을 no-replace로 발행한다.
2. `authority/stage2_2khz_physical.json`을 실제 binding SHA와 함께 검토·commit한다.
3. clean exact commit에서 기존 v1/v2 manifest를 history로 보존하며 v3를 발행한다.

```bash
.venv/bin/python scripts/data/build_elice_transfer_manifest.py \
  ...기존 strict P/S·recorded generation 인수... \
  --stage2-plant-binding results/<strict-stage2-capture>/plant_binding.json \
  --stage2-physical-authority authority/stage2_2khz_physical.json \
  --rotate-existing-transfer-sha256 <현재_manifest_sha256>
```

`--stage2-plant-binding`은 `--recorded-generation`과 함께만 사용할 수 있다.
따라서 parent 82세션만 든 v1을 Stage-2 v3로 잘못 승격할 수 없다. builder는
authority/binding을 실제로 다시 열어 production physical loader를 통과시킨 뒤,
`files[]`에 재해시한 SHA와 pointer SHA가 같은지 다시 검사한다.

## 새 256 GiB Elice 인스턴스 handoff

P/S PASS 및 해당 exact commit push 뒤에만 다음 순서를 사용한다.

1. 새 인스턴스에서 그 40자리 commit을 detached clean checkout으로 만든다.
2. `push_transfer_bundle.py`를 그 commit과 v3 manifest SHA로 실행한다. 전송 전에는
   local full semantic validator, 전송 후에는 원격 relative path/size/SHA 검증이 수행된다.
3. `bootstrap_all.sh --expected-commit <sha> --expected-transfer-manifest-sha256 <v3-sha> --no-update`를 실행한다.
   bootstrap은 v3를 combined-generation(101세션)으로 처리하지만 physical role을 일반
   recorded role로 약화하지 않는다.
4. Drive raw cache가 있더라도 cache receipt/decoder inventory와 v3 transfer SHA를 각각
   검증한다. cache는 physical P/S authority의 대체물이 아니다.
5. Elice에서 `scripts/data/issue_stage2_pretrain_data.py --plant-binding <v3의 role path> --expected-plant-binding-sha256 <v3의 role sha>`를 실행한다.
   issuer는 bootstrap receipt를 다시 열고 v3 role의 binding/authority 경로·SHA가 CLI
   artifact와 exact할 때만 public-data 후보를 발행한다.

실제 `authority/stage2_2khz_physical.json` 또는 physical artifact가 아직 없다면 이
절차를 실행하지 않는다. fixture는 테스트 전용이며 Elice canonical 전송/학습을 열지 않는다.
