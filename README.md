# 💰 채무 불이행 여부 예측 해커톤 : 불이행 징후를 찾아라!

![image](https://github.com/user-attachments/assets/955ad114-ffac-4399-a290-343765cf0ada)

🏆**최종 private score 1등**

## 📝 대회 개요
- 공식 사이트 : [링크](https://dacon.io/competitions/official/236450/overview/description) <- `클릭`
- 주최 : 데이콘
- 목적 : 금융 관련 데이터에 기반하여 채무 불이행 여부를 예측하는 AI 알고리즘을 개발
- 대회 기간 : 2025.02.03 ~ 2025.03.31
- 참여 기간 : 2025.03.05 ~ 2025.03.31

## 📁 데이터(train.csv)
| #   | Column                   | Non-Null Count   | Dtype    |
|-----|--------------------------|------------------|----------|
| 0   | UID                      | 10000 non-null   | object   |
| 1   | 주거 형태                 | 10000 non-null   | object   |
| 2   | 연간 소득                 | 10000 non-null   | float64  |
| 3   | 현재 직장 근속 연수       | 10000 non-null   | object   |
| 4   | 체납 세금 압류 횟수       | 10000 non-null   | float64  |
| 5   | 개설된 신용계좌 수        | 10000 non-null   | int64    |
| 6   | 신용 거래 연수            | 10000 non-null   | float64  |
| 7   | 최대 신용한도             | 10000 non-null   | float64  |
| 8   | 신용 문제 발생 횟수       | 10000 non-null   | int64    |
| 9   | 마지막 연체 이후 경과 개월 수 | 10000 non-null   | int64    |
| 10  | 개인 파산 횟수            | 10000 non-null   | int64    |
| 11  | 대출 목적                 | 10000 non-null   | object   |
| 12  | 대출 상환 기간            | 10000 non-null   | object   |
| 13  | 현재 대출 잔액            | 10000 non-null   | float64  |
| 14  | 현재 미상환 신용액        | 10000 non-null   | float64  |
| 15  | 월 상환 부채액            | 10000 non-null   | float64  |
| 16  | 신용 점수                 | 10000 non-null   | int64    |
| 17  | 채무 불이행 여부          | 10000 non-null   | int64    |

## 🔍 EDA
### 상관 관계
![image](https://github.com/user-attachments/assets/7f9068e1-0bee-485c-b94e-38f35e8fea02)


## ✂️ 전처리
- 파생변수 생성
- 대출목적, 개설된 신용계좌 수 drop
- 범주형 데이터 처리
- 이상치 처리(원저라이징, IQR)
- 로그 변환
- 결측값 처리(KNNImputer)
- 정규화(Standard Scaler)

## 	🤖 모델링
```python
sgd_model = SGDClassifier(loss='log_loss',
                             penalty='l2',
                             max_iter=1000,
                             random_state=42)
```

## 🎯 결과
### 📈 ROC-AUC
- Train Score: 0.6729
- Test Score : 0.6460
- Public Score : 0.6706
- Private Score : 0.6704(1등)

### 📊 모델 해석
#### SGDClassifier의 계수 (Coefficient) 기반 해석
| ![image](https://github.com/user-attachments/assets/f7d07da4-ab8d-4aad-ae62-4e892fafc47f) | ![image](https://github.com/user-attachments/assets/fb88f1ba-e983-4fc7-81ea-a5ed96d2957a) |
|---------------------------------------------------|--------------------------------------------------------|

선형 모델의 특성상, 각 변수의 계수(`coef_`)를 통해 해당 변수가 채무 불이행 가능성에 어떤 방향(긍정/부정)으로 작용했는지를 확인 가능
- 양의 계수: 해당 변수의 값이 클수록 불이행 가능성이 높아짐
- 음의 계수: 해당 변수의 값이 클수록 불이행 가능성이 낮아짐

```python
feature_importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
```

#### SHAP 
| ![image](https://github.com/user-attachments/assets/eef5be89-10b0-44f3-a241-6c28a4941710) | ![image](https://github.com/user-attachments/assets/4459bf4a-44cb-46be-8a25-36e5fc6de7e3) |
|---------------------------------------------------|--------------------------------------------------------|

SHAP는 각 예측값에 대해 개별 변수들이 얼마나 영향을 미쳤는지를 수치적으로 설명해줌

- SHAP value는 각 feature가 예측값(채무 불이행 확률)을 얼마나 올리거나 낮췄는지를 나타냅니다.
- summary plot (bar형): 전체 변수의 평균 영향력을 기준으로 중요도 순위를 시각화
- summary plot (dot형): 각 변수의 실제 값에 따라 예측값을 얼마나 밀어올렸는지(↑) 또는 낮췄는지(↓) 표현

주요 해석 결과:
- `연간 소득`이 높을수록 SHAP 값은 음수 → **불이행 가능성을 낮추는 방향으로 작용**
- `대출 상환 기간_장기 상환`, `개인 파산 횟수`, `현재 대출 잔액` 등의 변수는 **예측값을 상승시키는 주요 요인**

  
## 🧠 회고 및 배운 점
- 처음 참여해본 대회라 부족한 점이 많았지만, 좋은 결과를 얻
- 대회 끝난 후 코드 정리하다보니 오류가 발견됨. 오류들을 수정하니 성능이 떨어짐
- 다시 1등 순위의 족하는 성능을 만들기 위해 대회가 끝났지만, 전처리를 통해 성능을 비슷하게 만들음
- 자세한 느낀점 및 배운 내용 : [링크](https://accessible-riverbed-2b7.notion.site/1ca2f571f79c80f2beddcb08b03950de) <- `클릭`
