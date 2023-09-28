%AlexNet Transfer Learning

Alexnet
net = alexnet;
Access Layers
layers = net.Layers;
Train
Set up training data
rootFolder = '/Users/blaiseswartwood/Downloads/matlabdataprep/Scalogram/train';
categories = {'ALS','Normal','Myopathy'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');


numTrainFiles = 9000;
numValFiles = 1000;
[imdsTrain,imdsValidation, imdsTest] = splitEachLabel(imds,numTrainFiles, numValFiles, 'randomize');
imdsTrain.ReadFcn = @readFunctionTrain;
imdsValidation.ReadFcn = @readFunctionTrain;
imdsTest.ReadFcn = @readFunctionTrain;
Add Layers
layers = layers(1:end-3);


layers(end+1) = fullyConnectedLayer(64, 'Name', 'special_2');
layers(end+1) = reluLayer("Name","relu_add_1");
layers(end+1) = fullyConnectedLayer(3, 'Name', 'fc8_2 ');
layers(end+1) = softmaxLayer("Name","softmax");
layers(end+1) = classificationLayer("Name","classoutput")


layers
lgraph = layerGraph(layers)
plot(lgraph)
fpath = '/Users/blaiseswartwood/Desktop/Science Fair/CNN'
fname = ' Transfer Layers'
saveas(gcf, fullfile(fpath, fname), 'jpeg');
Fine-tune learning rates
layers(end-2).WeightLearnRateFactor = 10;
layers(end-2).WeightL2Factor = 1;
layers(end-2).BiasLearnRateFactor = 20;
layers(end-2).BiasL2Factor = 0;
training options
opts = trainingOptions('sgdm', ...
   'LearnRateSchedule', 'none',...
   'MaxEpochs', 25, ...
   'InitialLearnRate', .0001,...
   'Shuffle', 'every-epoch',...
   'ValidationData', imdsValidation,...
   'ValidationFrequency', 50, ...
   'Plots', 'training-progress', ...
   'MiniBatchSize', 128);
Train!
%addTrain
convnet = trainNetwork(imdsTrain, layers, opts);
plot(convnet)

Test
Test classifer
[labels,err_test] = classify(convnet, imdsTest, 'MiniBatchSize', 64);
Confusion Matrix and Accuracy
cm = confusionchart(imdsTest.Labels, labels)
cm.Title = 'Transfer Scalogram Confusion Matrix'
cm.NormalizedValues
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
fpath = '/Users/blaiseswartwood/Desktop/Science Fair/CNN'
fname = 'Test Transfer Scalogram'
saveas(gcf, fullfile(fpath, fname), 'jpeg');


confMat = confusionmat(imdsTest.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))
analyzeNetwork(convnet)

