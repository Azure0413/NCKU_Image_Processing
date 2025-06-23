# NCKU_Image_Processing

## Homework 1
(a) Use the average filter and the median filter, and compare their results.  
(b) Use the median filter to remove noise.  
(c) Please compare the results of (a) and (b) in view of performance.  
(d) Use the Sobel mask to sharp this Figure 2.  
(e) Use the Fourier transform to sharp this Figure 2.  
(f) Compare the results of (a) and (b) in view of performance.  

## Homework 2  
專案目標  
使用X光的影像分析是否有骨裂（Fracture）的形況，若有骨裂則顯示骨裂位置，若無則不須標註，最後要產生預測的Accuracy, Precision, Recall以及Bounding Box 的Mean IOU。  
專案方法  
由於圖像骨裂位置佔整體面積較小，因此若直接預測效果會不好（已實驗過），因此本專案先透過Yolov8l進行Segmentation，框出舟骨位置（Mean IOU = 0.92），接著針對舟骨位置進行Fracture Detection，使用的模型一樣是Yolov8l。  
