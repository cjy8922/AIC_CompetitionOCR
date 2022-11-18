# SW 중심대학 공동 AI 경진대회 (본선) - Optical Character Recoginition (OCR)
- 팀 경하예프 (경희대)
- 참가자: 김민성, 박수용, 이상민, 차준영, 한주혁
- Private Score (Accuracy): 92.44% (최종 10등)
- **발표 평가 및 최종 순위: 5등**

## Overview 
광학 문자 인식 Task로 주어진 데이터를 보고 어떤 문제가 있고, OCR 모델의 정확도를 향상시키기 위해선 이 문제들을 어떻게 해결해야 하는지 위주로 접근했습니다.

## Reference
- TPS-Resnet-BiLSTM-Attn: BAEK, Jeonghun, et al. What is wrong with scene text recognition model comparisons? dataset and model analysis. In: Proceedings of the IEEE/CVF international conference on computer vision. 2019. p. 4715-4723. 

(Official code: https://github.com/clovaai/deep-text-recognition-benchmark)
- SATRN: LEE, Junyeop, et al. On recognizing texts of arbitrary shapes with 2D self-attention. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2020. p. 546-547. 

**(Not official code, but we applied this code to train OCR model: https://github.com/Media-Smart/vedastr)**

## Library
<img src="https://img.shields.io/badge/lmdb-1.3.0-green"/> <img src="https://img.shields.io/badge/exrex-0.10.5-yellowgreen"/> <img src="https://img.shields.io/badge/nltk-3.7-yellowgreen"/> 

<img src="https://img.shields.io/badge/numpy-1.23.4-blue"/> <img src="https://img.shields.io/badge/opencv-4.6.0.66-blue"/> <img src="https://img.shields.io/badge/albumentations-1.3.0-blue"/> 

<img src="https://img.shields.io/badge/pytorch-1.11.0-red"/> <img src="https://img.shields.io/badge/torchvision-0.12.0-red"/>  ....

## Problem Solving Strategy
![슬라이드 1](/presentation/슬라이드1.PNG)
![슬라이드 2](/presentation/슬라이드2.PNG)
![슬라이드 3](/presentation/슬라이드3.PNG)
![슬라이드 4](/presentation/슬라이드4.PNG)
![슬라이드 5](/presentation/슬라이드5.PNG)
![슬라이드 6](/presentation/슬라이드6.PNG)
![슬라이드 7](/presentation/슬라이드7.PNG)
![슬라이드 8](/presentation/슬라이드8.PNG)
