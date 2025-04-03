# 채무 불이행 여부 예측 해커톤 : 불이행 징후를 찾아라!

![image](https://github.com/user-attachments/assets/955ad114-ffac-4399-a290-343765cf0ada)

`최종 private score 1등`

## 대회 개요
- 공식 사이트 : [링크](https://dacon.io/competitions/official/236450/overview/description)
- 주최 : 데이콘
- 목적 : 금융 관련 데이터에 기반하여 채무 불이행 여부를 예측하는 AI 알고리즘을 개발
- 대회 기간 : 2025.02.03 ~ 2025.03.31
- 참여 기간 : 2025.03.05 ~ 2025.03.31

## 데이터(train.csv)
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

## EDA
### 상관 관계
![image](https://github.com/user-attachments/assets/7f9068e1-0bee-485c-b94e-38f35e8fea02)


## 전처리
- 파생변수 생성
- 대출목적, 개설된 신용계좌 수 drop
- 범주형 데이터 처리
- 이상치 처리(원저라이징, IQR)
- 로그 변환
- 결측값 처리(KNNImputer)
- 정규화(Standard Scaler)

## 모델링
```python
sgd_model = make_pipeline(
    StandardScaler(),
    SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, random_state=42)
)
```

## 결과
- Train Score: 0.6729
- Test Score : 0.646
- Public Score : 0.6706
- Private Score : 0.6633(1등)
