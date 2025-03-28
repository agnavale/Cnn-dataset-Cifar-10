cifar -10 input dim = (32, 32, 3) <br>
classic cnn model. First conv layer with 64 kernals of dim= 3*3, stride = 1, padding = 1 <br>
layer1 output = (32,32,64) <br>
Second layer 128 kernals of dim = 3*3 stide = 1, padding= 1<br>
layer2 output = (32,32,128)
Third layer 256 kernals -----//------
layer3 output = (32,32,256)
Fourth layer maxpool kernal 2*2 and stride =2. Reduces each dimention by half
layer4 output = (16,16,128)


Model showed 80% accuracy.
