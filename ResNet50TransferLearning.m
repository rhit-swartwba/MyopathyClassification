%ResNet50

Resnet50
net = resnet50;
Access Layers
lgraph = layerGraph(net);
Train
Set up training data
rootFolder = '/Users/blaiseswartwood/Downloads/Dataprep2/Scalogram/train';
categories = {'Normal','Myopathy'};
imdsTrain = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imdsTrain.ReadFcn = @readFunctionTrain;


augmenter = imageDataAugmenter('RandXReflection',1)
imageSize = [224 224 3];
augimdsTrain = augmentedImageDatastore(imageSize, imdsTrain,'DataAugmentation',augmenter)


rootFolder = '/Users/blaiseswartwood/Downloads/Dataprep2/Scalogram/test';
imdsValidation = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imdsValidation.ReadFcn = @readFunctionTrain;
Add Layers
newFullyConnectedLayer =  fullyConnectedLayer(2, ...
       'Name','new_fc', ...
       'WeightLearnRateFactor',10, ...
       'BiasLearnRateFactor',10)
lgraph = replaceLayer(lgraph,'fc1000',newFullyConnectedLayer);
newClassificatonLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassificatonLayer);


plot(lgraph)
fpath = '/Users/blaiseswartwood/Desktop/Science Fair/CNN'
fname = ' Resnet50 Transfer Layers'
saveas(gcf, fullfile(fpath, fname), 'jpeg');
training options
checkpath = '/Users/blaiseswartwood/Downloads/emglab1/Blaise stuff/resnet'
opts = trainingOptions('adam', ...
   'MaxEpochs', 25, ...
   'Shuffle', 'every-epoch',...
   'Plots', 'training-progress', ...
   'CheckpointPath', checkpath,...
   'ValidationData', imdsValidation,...
   'ValidationFrequency', 50,...
   'Verbose', true,...
   'MiniBatchSize', 128);
Train
%addTrain
convnet = trainNetwork(augimdsTrain, lgraph, opts);
plot(convnet)

Test
Test classifer
rootFolder = '/Users/blaiseswartwood/Downloads/Dataprep2/Scalogram/final';
categories = {'Normal','Myopathy'};
imdsTest = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imdsTest.ReadFcn = @readFunctionTrain;


[labels,err_test] = classify(convnet, imdsTest, 'MiniBatchSize', 64);
Confusion Matrix and Accuracy
cm = confusionchart(imdsTest.Labels, labels)
cm.Title = 'ResNet50 Scalogram Confusion Matrix'
cm.NormalizedValues
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
fpath = '/Users/blaiseswartwood/Desktop/Science Fair/CNN'
fname = 'ResNet50 Transfer Scalogram CM'
saveas(gcf, fullfile(fpath, fname), 'jpeg');


confMat = confusionmat(imdsTest.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))
analyzeNetwork(convnet)

Filters
name = convnet.Layers(175).Name
channels = 1:2;
I = deepDreamImage(convnet,name,channels, 'NumIterations',100, ...
   'PyramidLevels',2);
figure
I = imtile(I,'ThumbnailSize',[250 250]);
imshow(I)
title(['Layer ',name,' Features'],'Interpreter','none')







