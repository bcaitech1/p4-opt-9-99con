# T1145 이동현
기존 Baseline 코드에 추가하거나 수정한 코드들이 포함되어 있습니다.   
/exp에 학습 로그들이 저장되어 있고 실제 제출했던 모델들만 남겨놨습니다. 
## 역할
model seacrh

## 작성 및 수정 사항
- 99tune.py
기존 AutoML을 통해 모델을 찾는 tune.py에 더 많은 hyperparameter를 고려하기 위한 코드로 수정하였습니다.

- 99tune_pre.py
수정한 99tune.py를 torchvision 모델들을 사용하기 위한 코드로 수정하였습니다.

- train_fine.py
학습 로그를 불러와 fine-tunning 하기 위해 기존 train 코드를 수정하여 작성하였습니다.

- src/model.py
torchvision 모델을 불러오기 위해 Model class를 일부 수정하였고, TVModel class로 torchvision 모델(shufflenetV2, mnasnet)을 불러올 수 있게 수정하였습니다.

- src/trainer.py
학습 로그를 저장할 때 macs를 확인할 수 있게 변경하였고, f1-score를 같이 확인할 수 있게 코드를 수정하였습니다.

- src/optim.py
AdamP와 SGDP를 추가하였습니다.

- src/loss.py
기존 코드에 Focal loss와 f1 loss를 추가하였습니다.