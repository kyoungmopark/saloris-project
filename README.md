# Smart_Parkinglot - 영상 데이터 기반의 지능형 주차 유도 시스템

---

## 프로젝트 소개

- 주제 : 영상 데이터 기반의 지능형 주차 유도 시스템 개발
- 과제 목표 및 내용 : 한국 패션센터 공영주차장에 차량 검지 및 추적 시스템을 설치하여 주차 현황 및 100% 무인 정산에 지원하고자 한다.
- 기대 효과 :

1. 무인 시스템 도입으로 인한 상주 인력 인건비 절감
2. 지능형 주차 유도 플랫폼을 제공하여 다른 주차장에 적용 가능
3. 주차장 이용자가 잔여 자리를 미리 파악하여 편의성 개선
4. 본 시스템을 도입 시, 공영주차장 통합관리(주차면 사용 수)이 가능하며, 공영주차장 사용자에게 잔여 자리를 제공하여 주차 편의성 증대를 기대할 수 있음

---

## 개발 기간

2023.07.03 ~ 2023.08.27 (8주)

7.3 ~ 7.9(1주차) 프로젝트 설계, 계획 수립, 개발환경 설정, 딥러닝 모델 검증  
7.10 ~ 7.16(2주차) 프로그램 기능 구현( 추적, 멀티쓰레드, 주차 여부 확인)  
7.17 ~ 7.23(3주차) 실험 환경에서 테스트 및 구현, 멘토링 세미나 참석
7.27 ~ 7.31(4주차) 모형 환경에서 테스트 및 구현, 실제 데이터 수집을 위한 영상 촬영
8.1 ~ 8.6(5주차) 촬영 데이터 기반의 프로그램 테스트 및 구현, 오류 수정
8.7 ~ 8.13(6주차) 촬영 데이터 기반의 프로그램 테스트 및 구현 ,오류 수정
8.14 ~ 8.20(7주차) SW 교육 (Netminer, SW 붐업 캠프)
8.21 ~ 8.27(8주차) 키오스크 앱 개발 및 개발 실습 내용 정리

---

## 개발자 소개

박경모 :

윤성원 :

김준영 :

---

## 기술 스택

개발 언어 : Python(Server), Flutter(Client)
프레임 워크 : YOLO, OpenCV, Pytorch, Firebase

---

## 프로젝트 아키텍쳐

---

## 주요 기능
### Real-Time Video Stitching
여러 개의 카메라에서 받아 온 영상을 한 개의 매끄러운(접합된) 영상으로 만들 수 있다면, 그 영상에 Object Detecting을 수행하여 주차장에 있는 차량을 관리 할 수 있을 것이라는 아이디어로 개발을 시작하게 되었다.
개발 언어: 파이썬
사용한 오픈소스 : https://github.com/OpenStitching/stitching(파이썬 이미지 스티칭 모듈)
기존 모듈을 그대로 사용했을 때 문제점은 매 프레임마다 새로운 특징점을 검출하고 이를 바탕으로 스티칭을 진행하여 전체 이미지의 크기의 변동이 발생하고 연산량이 높아 프레임 레이트가 저하된다는 점이였다.
이를 개선하여 Real-Time Video Stitching을 구현하기 위해서 첫 프레임 스티칭에 사용된 camera parameter를 캐시 메모리에 저장하고 Warp단계에서 인자로 전달하는 코드를 작성했다.
결과적으로 스티칭 된 영상의 크기가 변동하는 문제점을 해결했고 프레임 레이트 개선이 일정 부분 있었지만(2 camera 기준 4 FPS) 높은 연산 비용 문제로 6대의 CCTV 영상을 스티칭 하기엔 어려움이 있을 것 같아 다른 방법을 모색하기로 했다.

![image](https://github.com/kyoungmopark/saloris-project/assets/114475881/1ba0ff01-2cd1-45cf-a638-548494a3cdfc)
---
### ROI 프레임 생성
우리가 검지해야 하는 차량은 주차장 내에 있는 차량이지만 CCTV는 주차장 외의 영역(도로, 이웃 주차장, 건물 등)까지 촬영하는 문제점이 있었다.
이런 문제점을 해결하기 위해 관심 영역을 설정해서 Object Tracker에 전달하는 방법을 생각했고 Polygon(다각형) 형태로 프레임을 잘라서 새로운 프레임을 만드는 기능을 개발했다.

![image](https://github.com/kyoungmopark/saloris-project/assets/114475881/fc913624-d53c-402c-b8e0-784eda447b5e)

사용자가 CCTV 영상 캡처 이미지에 마우스 클릭으로 Polygon ROI를 만들고 해당 정보를 파일로 저장한다.
프로그램이 실행되면 Polygon 좌표를 불러와 ROI영역 외에는 검은색으로 마스킹 처리한 프레임을 만든 뒤 Object Tracker에게 전달한다.
이 결과, Tracker는 ROI 영역 내에 있는 차량만 검지하고 추적한다.

---

### 주차 확인 기능
기존에는 OpenCV 기반의 주차 감지 기능(https://github.com/olgarose/ParkingLot) 오픈소스를 사용해서 과제에 적용했다.
이 오픈소스의 작동원리는 마우스 클릭으로 주차 공간을 사각형 모양으로 설정하고 해당 영역 내부의 Laplacian값이 시간에 따라서 변하는지 확인하는 방식이다.

![image](https://github.com/kyoungmopark/saloris-project/assets/114475881/f25ff0eb-75d8-42d5-a187-4389b923aaca)

이 코드를 실제 환경에 적용해 보았을 때 높은 연산 비용 문제와 차량의 색상에 따른 인식 문제가 발생했다. 
이를 해결하기 위해 Object Tracker의 Bounding Box center 좌표가 주차 영역 내에 있는지 판별하는 간단한 원리를 통해 차량의 주차 여부를 판단하는 코드를 작성했고, 기존 오픈소스를 이로 대체했다.

![image](https://github.com/kyoungmopark/saloris-project/assets/114475881/d8f4287a-f72c-452c-8dbb-8273ae786943)
![image](https://github.com/kyoungmopark/saloris-project/assets/114475881/57718028-1bc8-433d-b8fe-97b2f9f28d5e)

형식 : “프로세스 ID” + “차량ID” 는 “주차영역 ID”에 주차중입니다.

### ROI 영역 객체 추적 & 멀티 프로세싱
멀티 프로세싱으로 각 영상을 입력 받고 ROI 영역에서만 Yolov5+ SORT를 실행하는 코드를 작성한 다음 기존에 개발한 주차 확인 기능과 결합했다.
실시간 영상 처리를 위해서 Yolov5는 Pytorch + GPU 가속화를 사용했고 Tracker는 SORT를 사용했다.
실험 환경 테스트
실행 화면 - 프로세스 사용률(4 camera)

![image](https://github.com/kyoungmopark/saloris-project/assets/114475881/b7bd59ae-479f-4e34-8698-bab8053330cb)

-흰 테두리 영역(ROI) 내에서만 객체를 검출, 추적 -초록색 영역은 빈 주차공간, 빨간색 영역은 사용중인 주차 공간
-Custom Weight 사용(yolov5s로 훈련시킴) \*사용한 데이터 셋 : https://universe.roboflow.com/class-5ftjh/vehicle_custom/dataset/2

실제 환경 테스트

![image](https://github.com/kyoungmopark/saloris-project/assets/114475881/67a7914b-7f9a-4696-b877-2d31d399aea2)

-실제 환경에서 영상 데이터 수집 후 프로그램 테스트 -프로세스 당 평균 3 FPS의 속도로 실행되는 것 확인
-Pre-trained Weight 사용 (Yolov5l, class 지정- car, truck)

---
### 주차장 키오스크 앱
100% 무인 정산을 위해서 이용자가 주차장에서 출차할 때 요금을 정산할 수 있는 키오스크 앱을 개발했다.

개발 언어  
-Flutter(앱),
-python(정보 전송 클라이언트)

키오스크 앱 시나리오 -클라이언트가 서버로 차량의 정보를 전송한다. (차량 이미지, 차량 번호, 입차 시간) -이용자가 자신의 차량 번호를 검색하고 정산한다. -결제가 완료되면 앱은 차량 데이터를 DB로 전송하고 자신의 저장 공간에서 삭제한다.

키오스크 앱 화면
<P align="center" width="100%" height="100%">
  <img src = "https://github.com/kyoungmopark/saloris-project/assets/114475881/7c530b48-cce8-468a-8cdf-593e8f20625b" width="23%" height="450px">
  <img src = "https://github.com/kyoungmopark/saloris-project/assets/114475881/acebc903-01f3-4d14-a44b-5dae54649d0d" width="23%" height="450px">
  <img src = "https://github.com/kyoungmopark/saloris-project/assets/114475881/4e73576e-74e2-4eba-adda-13cbddf6b8c7" width="23%" height="450px">
  <img src = "https://github.com/kyoungmopark/saloris-project/assets/114475881/9551d92a-0dc7-4c48-b55e-1d1db5710b15" width="23%" height="450px">
</P>
---

### TODO

한대의 차량이 카메라 간의 이동을 할 때 동일한 ID를 유지할 수 있도록 하는 기능을 완성하지 못했다.
이를 구현하기 위해서 이미지 스티칭, Polygon ROI로 카메라 별 관심 영역 설정의 방법을 시도했지만 큰 해결책이 되지 못했다.

SORT(Object Tracker)는 내부적으로 객체 검지에 실패했을 때 새로운 ID를 부여하도록 설계되어 있다. 우리의 프로젝트 주제에 맞게 SORT의 내부 코드를 수정하여 Re-ID 매커니즘을 변화시키는건 매우 복잡한 과정이라고 판단되었다.
따라서 프로젝트를 완성시키기 위해 향후에 더 개발해야 하는 부분은 Re-ID가 발생했을 때 이 데이터를 핸들링 하는 시스템이다.
Tracker의 ID와는 별도로 차량 객체에 ID를 부여하고 유지하며 Tracker에서 Re-ID가 발생했을 때 이것이 새로운 차량의 진입인지, 기존 차량의 단순 Re-ID인지(가림 현상, 카메라 흔들림에 의한) 프로세스(카메라) 간에 ID를 주고 받거나, 기존의 ID를 유지하도록 선택하는 시스템을 개발해야 할 것이다.
