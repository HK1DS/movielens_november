# 추가 요청 (11/18) — TO_GLOBAL `IKGR_lgcn_dyn` seed 2021, 2022

은선 님, 남은 크레딧으로 아래 두 개만 부탁드립니다. seed 2020은 진욱 PC에서 11/18 저녁에 돌리므로 겹치지 않게 빼 두었습니다.

| 할 일 | 예상 시간 | 비고 |
|---|---|---|
| TO_GLOBAL `IKGR_lgcn_dyn` seed 2021 | RTX 3090 기준 약 2.5~3.5시간 | 3060 Ti 실측: epoch당 8.2분, 30~40 epoch |
| TO_GLOBAL `IKGR_lgcn_dyn` seed 2022 | 〃 | |

## 실행 (서버, 레포 루트에서)

```bash
git fetch origin
git checkout additional        # team-mate 브랜치 + 이 문서와 실행 스크립트만 추가된 브랜치
bash scripts/additional_TO_GLOBAL_lgcn.sh
```

스크립트가 하는 일:
1. 평가 입력이 없으면 `data_bundle/`에서 복원 (`.inter`는 TO 묶음, intent KG는 TO_GLOBAL 묶음)
2. `scripts/check_meta_kg.py`로 메타 KG 확인 (다르면 멈춥니다)
3. `run_pipeline.py --k 30 --split TO_GLOBAL --steps E --seeds 2021,2022 --specs IKGR_lgcn_dyn`
4. 결과를 `results_additional_TO_GLOBAL.zip`으로 묶기 (결과 JSON + `per_user_TO_GLOBAL/IKGR_lgcn_dyn*`)

- Windows라면 3번 명령만 그대로 실행하신 뒤 `run_k30_TO_GLOBAL` 폴더의 결과 JSON과 `per_user_TO_GLOBAL`을 zip으로 보내주시면 됩니다.
- 크레딧이 하나만 될 것 같으면 `SEEDS=2021 bash scripts/additional_TO_GLOBAL_lgcn.sh`로 seed 2021만 돌려 주세요.
- 중간에 끊겨도 끝난 seed는 저장되어 있고, 다시 실행하면 끝난 seed는 건너뜁니다 (한 seed 안에서 이어 하기는 없습니다).

## 확인 부탁드릴 것

- 완료하신 결과에 TO_GLOBAL 2단계(`LightGCN`, `IKGR_lgcn_dyn_nointent`)가 있다면, 11/16 `git pull`(TO_GLOBAL 평가 기준 변경) **이후**에 시작하신 것인지 알려 주세요.
  이전 기준 결과는 진욱 쪽 병합 스크립트(`scripts/merge_results.py`)가 "evaluated differently"로 걸러 내고, 다시 돌려야 합니다.
- 결과 zip은 지금처럼 보내 주시면 진욱이 `scripts/merge_results.py`로 합칩니다.
