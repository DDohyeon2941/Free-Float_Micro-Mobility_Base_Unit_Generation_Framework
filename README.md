# Enhanced Base Unit Generation Framework for Micro-Mobility Demand Prediction  

## 📌 Overview  
This project presents an **enhanced base unit generation framework** for demand prediction in **free-floating micro-mobility systems**. The method aims to **mitigate the Modifiable Areal Unit Problem (MAUP)**, which significantly impacts spatial data modeling, by introducing a **clustering-based approach** for defining areal units.  

The framework leverages **spatiotemporal clustering** to refine the areal unit definitions, ensuring that they align with real-world demand patterns while improving **prediction performance** in micro-mobility demand forecasting. The approach is validated using **shared e-scooter datasets from Kansas City and Minneapolis**.  


### 📂 **디렉토리 설명**
- **Kansas/** 및 **minneapolis/**: 각각 캔자스 시티와 미니애폴리스를 위한 데이터 및 분석 코드가 포함된 디렉토리입니다.
  - **analysis/**: 분석 및 시각화 관련 코드와 데이터가 포함되어 있습니다.
  - **dataset/**: 원본 데이터 및 전처리된 데이터가 포함된 디렉토리입니다.
  - **experiment/**: 실험 관련 코드와 설정이 포함된 디렉토리입니다.
  - **results/**: 실험 결과 및 분석 결과를 저장하는 디렉토리입니다.
  - **run_scripts/** (Kansas만 존재): 실행 관련 스크립트가 포함된 디렉토리입니다.
  - **spatial_units/**: 공간 단위 관련 데이터 및 분석 코드가 포함된 디렉토리입니다.
  - **utils/**: 공통적으로 사용되는 유틸리티 함수 및 보조 코드가 포함된 디렉토리입니다.
```
Free-Float_Micro-Mobility_Base_Unit_Generation_Framework/
├── Kansas/
│   ├── analysis/          # 분석 관련 코드 및 데이터
│   ├── dataset/           # 캔자스 시티 데이터셋
│   ├── experiment/        # 실험 관련 코드 및 설정
│   ├── results/           # 실험 결과 저장
│   ├── run_scripts/       # 실행 스크립트
│   ├── spatial_units/     # 공간 단위 관련 데이터 및 코드
│   ├── utils/             # 유틸리티 함수 및 공통 코드
│
├── minneapolis/
│   ├── analysis/          # 분석 관련 코드 및 데이터
│   ├── dataset/           # 미니애폴리스 데이터셋
│   ├── experiment/        # 실험 관련 코드 및 설정
│   ├── results/           # 실험 결과 저장
│   ├── spatial_units/     # 공간 단위 관련 데이터 및 코드
│   ├── utils/             # 유틸리티 함수 및 공통 코드
│
├── .gitattributes         # Git 속성 파일
├── README.md              # 프로젝트 소개 및 가이드
├── maup.yml               # MAUP 관련 설정 파일
└── 예측성능_20240816.xlsx # 예측 성능 결과 파일
```

이 내용을 **README.md**에 추가하면 프로젝트 구조를 한눈에 파악하기 쉽고, 협업 및 유지보수에도 도움이 될 것입니다. 🚀
## 🚀 Features  
- **MAUP-Aware Areal Unit Definition**: Generates dynamic spatial units that mitigate biases caused by arbitrary grid-based divisions.  
- **Spatiotemporal Clustering**: Merges areas with similar demand patterns while maintaining geographic continuity.  
- **Improved Demand Prediction**: Stabilizes the target distribution for better forecasting accuracy.  
- **Validated on Real-World Data**: Evaluated using **Kansas City and Minneapolis e-scooter datasets**.  

## 🔧 Methodology  
The framework consists of three main steps:  

1. **Base Grid Generation**  
   - The study first divides the city into **small base grids** and assigns **central points** based on actual rental and return patterns.  

2. **Main Grid Generation**  
   - Base grids with sufficient rental records are classified as **main grids**, while others are **support grids**.  
   - Support grids are merged into the nearest main grids to preserve **temporal similarity and spatial connectivity**.  

3. **Base Unit Generation**  
   - Main grids are **clustered using hierarchical agglomerative clustering**, incorporating both **temporal similarity** and **spatial proximity**.  
   - This results in **adaptive areal units** that mitigate extreme values and improve demand prediction.  

## 📊 Experimental Results  
The proposed framework was **compared against traditional grid-based methods** (250m, 500m, and 750m grid widths) using **various predictive models**. Key findings include:  
- **Reduction in extreme demand values**: The framework stabilizes the distribution, reducing both excessive zero-demand areas and extreme rental spikes.  
- **Improved prediction performance**: Particularly in spatial units that previously suffered from MAUP-related bias.  
- **Validation across different cities**: The framework's effectiveness is demonstrated across Kansas City and Minneapolis, showcasing adaptability to different urban environments.  

## **Contributing**
We welcome contributions to this project. Please submit pull requests or open issues for any bugs or enhancements.

## **Citation**
For more details, please refer to the full paper on [IET Intelligent Transport Systems](https://doi.org/10.1049/itr2.12596).
