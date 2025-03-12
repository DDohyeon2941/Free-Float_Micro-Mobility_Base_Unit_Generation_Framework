# Enhanced Base Unit Generation Framework for Micro-Mobility Demand Prediction  

## 📌 Overview  
This project presents an **enhanced base unit generation framework** for demand prediction in **free-floating micro-mobility systems**. The method aims to **mitigate the Modifiable Areal Unit Problem (MAUP)**, which significantly impacts spatial data modeling, by introducing a **clustering-based approach** for defining areal units.  

The framework leverages **spatiotemporal clustering** to refine the areal unit definitions, ensuring that they align with real-world demand patterns while improving **prediction performance** in micro-mobility demand forecasting. The approach is validated using **shared e-scooter datasets from Kansas City and Minneapolis**.  

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
