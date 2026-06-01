# 전처리v2 + Hydro-MAST (여주보 수위 AI · 전달 패키지)

**사용 설명:** [`사용설명서.txt`](사용설명서.txt)  
**압축 전달:** `압축만들기.bat` (전처리v2대안AI + yeoju 원천)

## 재실행

```powershell
.\.venv\Scripts\python preprocess_v2.py
.\.venv\Scripts\python train_hydro_mast.py
.\.venv\Scripts\python validate_hydro_mast.py
```

## 주요 산출

| 경로 | 용도 |
|------|------|
| `data/features_v2_train.csv` | 2년 학습 |
| `models/hydro_mast_v2.pt` | Hydro-MAST |
| `docs/preprocess_v2_spec.json` | 연동 스펙 |

원천: zip과 함께 푼 `yeoju_ai_handoff_2024_2025_KST/` (이 폴더와 형제 위치)
