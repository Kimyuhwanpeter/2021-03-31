# 2021-03-31
* 현재는 90 feature vector에 [9,10]으로 reshape하여 9는 나이 세대, 10은 세부 나이정보로 나누고 loss를 만들었다.
* 9세대는 one-hot encoding 을 이용하여 softmax cross entropy를 이용, 10 세부 나이는 CORAL에 나온 label 기법으로 처리한 뒤 sigmoid cross entropy를 사용하였다.
* *Colab에서 우선 실험중이며, learning rate schdule을 사용하여 학습하는것도 고려중이다.*
