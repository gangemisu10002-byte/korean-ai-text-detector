# 프로젝트 상세 설명서 — Korean AI Text Detector

## 1. 프로젝트 한눈에 보기

이 프로젝트는 한국어 텍스트가 **사람(Human)이 작성한 텍스트인지, AI가 생성한 텍스트인지** 분류하는 경량 머신러닝 프로토타입이다.

대형 언어모델을 새로 학습하는 대신 다음과 같은 전통적인 머신러닝 파이프라인을 사용한다.

```text
한국어 텍스트
    ↓
데이터 전처리
    ↓
TF-IDF 특징 추출
    ├─ Word TF-IDF
    └─ Character TF-IDF
    ↓
Logistic Regression
    ↓
Human / AI 분류
```

추가 실험으로 한국어 형태소 분석기 **Kiwi**에서 얻은 형태소/품사 정보를 특징으로 추가한 모델도 구축했다.

```text
한국어 텍스트
    ↓
Word TF-IDF
    +
Character TF-IDF
    +
Kiwi 형태소/품사 TF-IDF
    ↓
Logistic Regression
    ↓
Human / AI 분류
```

---

## 2. 이 프로젝트를 만든 이유

LLM의 발전으로 사람이 작성한 글과 AI가 생성한 글을 구별하는 문제가 중요해지고 있다.

특히 한국어는 영어와 달리 형태소 구조가 풍부하고 조사와 어미가 발달했으며, 띄어쓰기와 문장 구성에서 고유한 특징을 가진다. KatFishNet 연구 역시 이러한 한국어의 언어적 특성을 고려할 필요성을 강조한다.

원 연구인 **KatFishNet**은 한국어 AI 생성 텍스트 탐지를 위한 KatFish benchmark dataset과 한국어 특화 탐지 방법을 제안했다. 논문은 ACL 2025에 발표되었다.

출처:
- Park, Shinwoo; Kim, Shubin; Kim, Do-Kyung; Han, Yo-Sub. (2025).
- *KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis.*
- Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025), pp. 21189–21222.
- DOI: 10.18653/v1/2025.acl-long.1030

원 논문:
https://aclanthology.org/2025.acl-long.1030/

공식 코드/데이터 저장소:
https://github.com/Shinwoo-Park/katfishnet

---

## 3. 데이터셋 출처

이 프로젝트는 원 연구에서 공개한 **KatFish 데이터셋**을 사용한다.

ACL 논문에 따르면 KatFish는 한국어 AI 생성 텍스트 탐지를 위한 benchmark dataset이며, **세 가지 장르**의 텍스트를 포함한다.

- Essay
- Poetry
- Paper Abstract

또한 인간 작성 텍스트와 여러 LLM이 생성한 텍스트를 포함한다.

공식 저장소에서는 다음과 같은 데이터 파일을 확인할 수 있다.

```text
katfish_dataset/
├── essay.jsonl
├── poetry.jsonl
└── abstract.jsonl
```

### 중요한 출처 구분

이 프로젝트가 KatFish 데이터셋을 직접 만든 것은 아니다.

따라서 다음을 구분한다.

```text
원 연구자
  ↓
KatFish 데이터셋
  ↓
이 프로젝트에서 공개 데이터셋을 다운로드
  ↓
별도의 전처리 및 경량 분류 실험
  ↓
Base Model / Kiwi Model
```

데이터셋 및 원 연구에 대한 권리는 원 저작자 및 해당 저장소의 조건을 따른다.

---

## 4. 원 연구 KatFishNet과 이 프로젝트의 관계

이 프로젝트는 **KatFishNet 공식 구현이 아니다.**

원 연구에서는 한국어의 언어적 특성을 이용해 다음과 같은 특징을 분석했다.

1. Word spacing patterns
2. POS n-gram diversity
3. Comma usage patterns

그리고 이러한 언어적 특징을 이용해 AI 생성 텍스트를 탐지하는 KatFishNet 방법을 제안했다.

반면 이 프로젝트에서는 발표/교육용으로 더 가벼운 구현을 목표로 하여 다음 방법을 별도로 구성했다.

```text
Base Model
Word TF-IDF
+
Character TF-IDF
+
Logistic Regression
```

그리고 한국어 형태소 정보를 추가하는 실험으로:

```text
Kiwi Model
Word TF-IDF
+
Character TF-IDF
+
Kiwi linguistic features
+
Logistic Regression
```

을 구성했다.

따라서 이 저장소의 코드를 **KatFishNet의 공식 코드라고 표현하면 안 된다.**

---

# 5. 모델 1 — Base Model

## 5.1 TF-IDF란?

TF-IDF는 텍스트에서 특정 단어 또는 문자 패턴이 얼마나 중요한지를 수치로 표현하는 방법이다.

TF는 특정 문서에서 해당 feature가 얼마나 등장하는지를 반영하고, IDF는 전체 문서에서 흔하지 않은 feature에 더 높은 가중치를 주는 방식이다.

쉽게 말하면:

> "이 텍스트를 다른 텍스트와 구별하는 데 도움이 되는 단어/문자 패턴은 무엇인가?"

를 숫자로 바꾸는 과정이다.

---

## 5.2 Word TF-IDF

Word TF-IDF는 단어 단위의 특징을 추출한다.

예를 들어 텍스트가:

```text
한국어 인공지능 연구를 진행했다.
```

라면 단어 수준에서 다음과 같은 feature가 만들어질 수 있다.

```text
한국어
인공지능
연구
진행
...
```

이를 숫자 벡터로 변환하여 머신러닝 모델에 입력한다.

---

## 5.3 Character TF-IDF

Character TF-IDF는 단어보다 작은 문자 단위의 패턴을 활용한다.

한국어에서는 형태 변화나 어미, 조사 등의 차이가 문자 패턴에 나타날 수 있기 때문에 단어 TF-IDF와 다른 정보를 제공할 수 있다.

따라서 이 프로젝트에서는:

```text
Word TF-IDF
+
Character TF-IDF
```

를 결합한다.

---

# 6. 모델 2 — Kiwi Model

## 6.1 Kiwi란?

Kiwi는 한국어 형태소 분석 기능을 제공하는 도구이다.

이 프로젝트에서는 Kiwi 자체를 AI 탐지 모델로 사용하는 것이 아니다.

Kiwi를 이용해 텍스트의 형태소/품사 정보를 얻고, 그 정보를 추가 feature로 사용한다.

---

## 6.2 Kiwi Feature

예를 들어 문장을 형태소 분석하면 단어 자체뿐 아니라 품사 정보를 이용할 수 있다.

개념적으로:

```text
텍스트
 ↓
형태소 분석
 ↓
형태소 + 품사
 ↓
feature sequence
 ↓
TF-IDF
```

과 같은 방식으로 사용할 수 있다.

이렇게 하면 단순히 "어떤 단어가 등장했는가"뿐 아니라 **어떤 언어적 단위와 품사 패턴이 등장하는가**를 모델이 활용할 수 있다.

---

# 7. 두 모델의 차이

| 구분 | Base Model | Kiwi Model |
|---|---|---|
| Word TF-IDF | O | O |
| Character TF-IDF | O | O |
| Kiwi 형태소/품사 Feature | X | O |
| 분류기 | Logistic Regression | Logistic Regression |
| 목적 | 기본 텍스트 패턴 비교 | 한국어 언어정보 추가 효과 비교 |
| 계산량 | 상대적으로 낮음 | Base보다 높음 |

핵심 실험 질문은 다음과 같다.

> **한국어 형태소/품사 정보를 추가하면 기본 TF-IDF 모델보다 AI/Human 분류에 도움이 되는가?**

---

# 8. Logistic Regression을 사용한 이유

Logistic Regression은 이진 분류에 적합한 가벼운 머신러닝 알고리즘이다.

이 프로젝트의 목표가 대형 모델을 학습하는 것이 아니라 **CPU/Colab에서도 빠르게 실행되는 프로토타입을 만드는 것**이었기 때문에 선택했다.

장점:

- CPU에서 실행하기 쉬움
- TF-IDF의 고차원 sparse feature와 함께 사용하기 좋음
- 학습 속도가 비교적 빠름
- 모델 구조가 비교적 단순함
- 결과를 설명하기 쉬움

---

# 9. 데이터 전처리

Notebook에서는 JSONL 파일을 읽어 데이터프레임으로 변환한 뒤 텍스트와 label을 확인한다.

기본적으로 중요한 확인 항목은:

```text
text
label
source_type
```

등이다.

또한 데이터에 포함될 수 있는 불필요한 문자와 공백 등을 정리한다.

예:

```python
df["text"] = (
    df["text"]
    .astype(str)
    .str.replace("​", " ", regex=False)
)
```

그리고 label이 한 종류만 존재하는지 확인한다.

### 왜 이 검사가 중요한가?

Logistic Regression이나 SVM 같은 이진 분류기는 학습 데이터에 최소 두 개의 class가 있어야 한다.

예를 들어:

```text
label
0    470
```

처럼 한 class만 존재하면 학습할 수 없다.

반드시:

```text
label
0    ...
1    ...
```

처럼 두 class가 존재하는지 확인해야 한다.

---

# 10. Train / Validation / Test

데이터를 학습과 평가에 나누어 사용한다.

개념적으로:

```text
전체 데이터
   ↓
Train
   ├─ 모델 학습
   ↓
Validation
   ├─ 설정/비교
   ↓
Test
   └─ 최종 성능 확인
```

중요한 원칙은 **테스트 데이터가 모델 학습에 사용되지 않도록 하는 것**이다.

---

# 11. 모델 평가

주요 평가 지표는 다음과 같다.

## Accuracy

전체 예측 중 맞힌 비율이다.

```text
Accuracy =
맞힌 샘플 수 / 전체 샘플 수
```

## Precision

AI라고 예측한 샘플 중 실제 AI인 비율을 나타낸다.

## Recall

실제 AI 샘플 중 모델이 AI라고 찾아낸 비율이다.

## F1-score

Precision과 Recall의 조화평균이다.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

## ROC-AUC

분류기의 순위/판별 능력을 평가하는 대표적인 지표다.

특히 단순히 하나의 threshold에서 맞혔는지만 보는 것보다 다양한 threshold에서 분류 능력을 평가하는 데 유용하다.

---

# 12. 코드 실행 흐름

Notebook 전체는 다음 흐름으로 이해하면 된다.

```text
Cell 1
환경/라이브러리 준비
        ↓
Cell 2~
KatFishNet 저장소 다운로드
        ↓
JSONL 탐색
        ↓
essay / poetry / abstract 로딩
        ↓
데이터 검증 및 전처리
        ↓
Train / Validation / Test 분리
        ↓
Base TF-IDF feature 생성
        ↓
Base Logistic Regression 학습
        ↓
Kiwi 형태소 feature 생성
        ↓
Kiwi Logistic Regression 학습
        ↓
성능 평가
        ↓
Base vs Kiwi 비교
        ↓
모델 저장
        ↓
Gradio 입력기
```

---

# 13. 모델 저장 과정

학습이 완료된 모델은 다시 사용할 수 있도록 joblib 형태로 저장한다.

중요한 점은 Kiwi 객체 자체를 저장하지 않는 것이다.

이전에 발생했던:

```text
TypeError:
cannot pickle 'Kiwi' object
```

오류는 Kiwi 객체를 직접 joblib으로 직렬화하려고 했기 때문에 발생했다.

따라서 모델 bundle에는 실제 추론에 필요한 직렬화 가능한 객체만 저장하고, Kiwi 분석기는 실행 시 다시 생성하는 구조가 적절하다.

개념적으로:

```text
저장
├── TF-IDF vectorizer
├── classifier
└── 기타 필요한 설정
```

으로 구성한다.

---

# 14. 입력기

학습이 끝나면 사용자가 직접 한국어 문장을 입력할 수 있는 인터페이스를 사용할 수 있다.

입력:

```text
사용자가 한국어 텍스트 입력
```

처리:

```text
입력
 ↓
Base Model
 ↓
Human / AI prediction

입력
 ↓
Kiwi Model
 ↓
Human / AI prediction
```

두 모델을 동시에 보여주면:

```text
Base Model
예측: AI
확률: ...

Kiwi Model
예측: Human
확률: ...
```

처럼 비교할 수 있다.

이 구조의 목적은 **모델 간 판단 차이를 관찰하는 것**이다.

---

# 15. 이 프로젝트에서 얻은 것

## 15.1 데이터셋 구축 경험

공개 데이터셋을 단순히 다운로드하는 것에서 끝나지 않고:

```text
Repository
→ JSONL
→ 데이터 구조 확인
→ label 검증
→ 전처리
→ 학습용 데이터
```

로 연결하는 과정을 경험했다.

## 15.2 머신러닝 전체 pipeline 경험

모델 하나를 호출하는 것보다 중요한 것은 전체 pipeline이다.

```text
Data
→ Preprocessing
→ Feature Engineering
→ Training
→ Evaluation
→ Serialization
→ Inference
```

을 직접 연결했다.

## 15.3 Feature Engineering의 중요성

같은 Logistic Regression을 사용해도 어떤 feature를 넣느냐에 따라 모델이 사용하는 정보가 달라진다.

이번 프로젝트에서는:

```text
Word
Character
Kiwi linguistic feature
```

를 비교함으로써 feature engineering의 역할을 확인했다.

## 15.4 한국어 NLP 경험

한국어 텍스트에서는 단순 단어 빈도뿐 아니라 형태소와 품사 등의 정보가 중요할 수 있다는 점을 실험했다.

## 15.5 가벼운 모델 설계

대형 Transformer 모델을 직접 학습하지 않고도 CPU/Colab 환경에서 실행 가능한 텍스트 분류 시스템을 만들었다.

---

# 16. 결과값을 발표할 때 주의할 점

실험 결과는 **실제로 Notebook을 실행한 값만 기록해야 한다.**

예를 들어 다음과 같이 표를 만들 수 있다.

| Metric | Base | Kiwi |
|---|---:|---:|
| Accuracy | 실제 실행값 | 실제 실행값 |
| Precision | 실제 실행값 | 실제 실행값 |
| Recall | 실제 실행값 | 실제 실행값 |
| F1 | 실제 실행값 | 실제 실행값 |
| ROC-AUC | 실제 실행값 | 실제 실행값 |

실행하지 않은 숫자를 임의로 넣으면 안 된다.

---

# 17. 원 논문의 결과와 내 실험 결과를 혼동하면 안 되는 이유

원 연구의 KatFishNet 성능과 이 프로젝트의 Base/Kiwi 모델 성능은 같은 실험이 아니다.

원 논문은 자체적인 KatFishNet 방법과 실험 설정을 사용한다.

이 프로젝트는:

```text
Word TF-IDF
+
Character TF-IDF
+
선택적으로 Kiwi feature
+
Logistic Regression
```

이라는 별도의 경량 구현이다.

따라서 발표에서는:

> "KatFishNet 논문의 성능을 재현했다."

라고 표현하기보다는,

> "KatFish 데이터셋을 활용하여 별도의 경량 분류 모델을 구현하고 Base 모델과 Kiwi feature 추가 모델을 비교했다."

라고 표현하는 것이 정확하다.

---

# 18. 프로젝트의 한계

이 모델은 텍스트의 실제 작성자를 확정적으로 판정하는 도구가 아니다.

모델의 출력은:

> "학습된 데이터와 특징을 기준으로 이 텍스트가 어느 class에 가까운가"

를 나타내는 분류 결과다.

따라서 다음 상황에서 성능이 달라질 수 있다.

- 학습 데이터와 다른 장르
- 학습에 사용하지 않은 LLM
- 사람이 AI처럼 작성한 글
- AI가 사람처럼 수정한 글
- 매우 짧은 텍스트
- 새로운 도메인의 텍스트

따라서 교육/연구 목적의 프로토타입으로 해석하는 것이 적절하다.

---

# 19. 표절 및 출처 관련 원칙

이 프로젝트에서는 원 연구와 내가 직접 구현한 부분을 구분한다.

### 원 연구에서 가져온 것

- KatFish 데이터셋
- KatFishNet 연구의 아이디어/배경에 대한 참고
- 원 논문 정보

### 이 프로젝트에서 직접 구현한 것

- 데이터 다운로드/로딩 pipeline
- 데이터 검증 및 전처리
- Word/Character TF-IDF pipeline
- Logistic Regression 분류 pipeline
- Kiwi feature를 추가한 비교 모델
- 모델 저장 구조
- 두 모델을 비교하는 입력기
- 발표/실험용 프로젝트 구조

따라서 GitHub README와 NOTICE에 원 연구 및 데이터셋 출처를 명시한다.

---

# 20. 참고문헌

### Primary Reference

Park, Shinwoo, Shubin Kim, Do-Kyung Kim, and Yo-Sub Han. 2025.

**KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis.**

Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 21189–21222.

DOI: 10.18653/v1/2025.acl-long.1030

Paper:
https://aclanthology.org/2025.acl-long.1030/

Official repository:
https://github.com/Shinwoo-Park/katfishnet

### Libraries

본 프로젝트는 다음과 같은 오픈소스 라이브러리를 사용한다.

- scikit-learn — 머신러닝 및 TF-IDF
- pandas — 데이터 처리
- NumPy / SciPy — 수치 계산
- joblib — 모델 저장
- kiwipiepy — 한국어 형태소 분석
- Gradio — 입력 인터페이스
- Jupyter / IPython — Notebook 실행

각 라이브러리의 라이선스는 해당 프로젝트의 라이선스를 따른다.

---

