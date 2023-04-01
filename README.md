# RNN을 활용한 MNIST Classificatioon 구현

* 이미지는 Sequence한 데이터는 아니지만 LOW별로 따지고 보면 순서의 정보가 있음

![스크린샷 2023-04-01 오전 10 12 14](https://user-images.githubusercontent.com/84004919/229258471-e4a676ba-6808-4897-bc97-cc0b16b4ce9a.png)

* 입력이 한번에 들어와요
* 내부의 hidden state, cell state 크기까지 다 정해줘야대!
* 각 time stemp마다 결과가 나오지만 마지막 time stemp의 최종 값만 산출
* (batch size, 1, Hidden size*2 ) <- 정방향 + 역방향 수행하기때문에 hs*2

![스크린샷 2023-04-01 오전 10 20 06](https://user-images.githubusercontent.com/84004919/229258818-d22461f0-8e75-4aae-836f-43d35a432aac.png)

* mnist_classification/models/rnn_model.py
  - batch first = true
  - LSTM기본 = (time step, batch size, vector size ) == (28, bs, 28 )
  - batch first = true -> (batch size, time step, vector size ) == (bs, 28, 28 )
  ![스크린샷 2023-04-01 오전 10 27 39](https://user-images.githubusercontent.com/84004919/229259202-de8511ff-9dcc-46a7-ba9d-9926d9b5bbe5.png)

  
