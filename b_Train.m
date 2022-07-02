clc;
clear;
close all;

totalTeste = 0;
totalTotal = 0;
neuronios = 10;

camadas = [neuronios];
list = dir('Photos/train/**/*.png');

filenames = string({list.folder}) + '/'+string({list.name});
str = filenames;
i=1;
count = 0;
for st = str
    count = count +1;
end


img_res = [28 28]; %Size of image base to use
poligonos = zeros(img_res(1)*img_res(2)*3,count); % Initialize the matrix with zeros
for st = str
    numChar = strfind(st,"."); %Number of char in image name before the point
    S = extractBefore(st, numChar); % extract only the name without extension
    numChar1 = strfind(S,"/"); %Number of char in image name before the /
    St2 = extractBefore(S, numChar1); % extract only the name
    numChar2 = strfind(S,"/"); %Number of char in image name before the /
    St3 = extractAfter(S, numChar2); % extract only the name

    fileStList = St2+'\'+St3+'.png';

    I = imread(fileStList); %access image

    I = imresize(I,img_res); %resize or image

    BinImage = imbinarize(I);

    poligonos(:,i) = reshape(BinImage, 1, []);

    fileout = "Out/Binary/" + St3 + ".bin";
    %open the file
    fileID = fopen (fileout, 'w'); %creates bin file

    %image conversion to black and white
    %BW = im2bw(I, 0.4);

    %write into the file
    fprintf(fileID, '%d', poligonos(:,i)); %print to file


    %show comparisons
    %figure;
    %imshowpair(I, BW, 'montage') %Shows image

    i=i+1;

end

vec1 = repelem(1, 50);
vec2 = repelem(2, 50);
vec3 = repelem(3, 50);
vec4 = repelem(4, 50);
vec5 = repelem(5, 50);
vec6 = repelem(6, 50);

letrasTarget = [vec1, vec2, vec3, vec4, vec5, vec6];
letrasTarget = onehotencode(letrasTarget,1,'ClassNames',1:6);

% CEATE AND CONFIGURE the neural network
% Fill: Number od hidden layers and nodes by hidden layer
net = feedforwardnet(camadas);
% Fill: Type of trainning function: {'trainlm', 'trainbfg', traingd, traingdx'}
net.trainFcn = 'traingdx';
% Fill: Activation functions for the hidden and output layers; {'purelin', 'logsig', 'tansig'}
net.layers{end}.transferFcn = 'tansig';
net.layers{1:end-1}.transferFcn = 'purelin';
% Fill: Split of the sets examples training, validation and test
net.divideFcn = 'dividerand';
dataSegTrain = 0.40;
dataValRatio = 0.30;
dataTestRatio = 0.30;
net.divideParam.trainRatio = dataSegTrain;
net.divideParam.valRatio = dataValRatio;
net.divideParam.testRatio = dataTestRatio;
net.trainParam.epochs = 1000; %%TODO CHANGE TO 100 or 1000


[net,tr] = train(net, poligonos, letrasTarget);
out = net(poligonos);

disp(tr);

r = 0;
for i=1:size(out,2)
    [a, b] = max(out(:,i));
    [c, d] = max(letrasTarget(:,i));
    if b == d
        r = r+1;
    end

end

TTargets = letrasTarget(tr.testInd);
out_test = (out(tr.testInd)>0.5);
accuracyTotal = sum(out_test == TTargets)/length(tr.testInd);
fprintf('Precisao total = %f\n', accuracyTotal*100);

TInput = poligonos(:, tr.testInd);
TTargetss = letrasTarget(:, tr.testInd);


out = sim(net, TInput);

r=0;
for i=1:size(tr.testInd,2)               
    [a, b] = max(out(:,i));          
    [c, d] = max(TTargetss(:,i));  
    if b == d                       
        r = r+1;
    end
end

accuracy = r/size(tr.testInd,2);
fprintf('Precisao teste = %f\n', accuracy*100);
plotconfusion(TTargetss, out);


if(accuracyTotal ~= 1)
    numChar = strfind(num2str(round(accuracyTotal, 2)),'.'); 
    S1 = extractBefore(num2str(round(accuracyTotal, 2)), numChar); 
    S2 = extractAfter(num2str(round(accuracyTotal, 2)), numChar); 
else
    S1 = 1;
    S2 = 0;
end
strLayers = num2str(size(camadas));
strLayersFinal = extractAfter(strLayers, 3);

tituloRede = "Out/Redes/TrainPolig1_" + net.trainFcn+ "_" + S1 + "_" + S2 + "_" + strLayersFinal + "_" + num2str(neuronios) + "_" + "T_" + num2str(dataTestRatio)+ "_" + "V_" + num2str(dataValRatio) + "_" + "Tr_" + num2str(dataSegTrain) +".mat";
tituloPerformance = "Out/Performance/Start_Performance_" + net.trainFcn+ "_" + S1 + "_" + S2 + "_" + strLayersFinal + "_" + num2str(neuronios) + "_" + "T_" + num2str(dataTestRatio)+ "_" + "V_" + num2str(dataValRatio) + "_" + "Tr_" + num2str(dataSegTrain) +".fig";
tituloConfusion= "Out/Confusion/Start_Confusion_" + net.trainFcn+ "_" + S1 + "_" + S2 + "_" + strLayersFinal + "_" + num2str(neuronios) + "_" + "T_" + num2str(dataTestRatio)+ "_" + "V_" + num2str(dataValRatio) + "_" + "Tr_" + num2str(dataSegTrain) + ".fig";
tituloTrainState= "Out/TrainState/Train_TrainState_" + net.trainFcn+ "_" + S1 + "_" + S2 + "_" + strLayersFinal + "_" + num2str(neuronios) + "_" + "T_" + num2str(dataTestRatio)+ "_" + "V_" + num2str(dataValRatio) + "_" + "Tr_" + num2str(dataSegTrain) +".fig";

savefig(tituloConfusion);
set(findobj(gca,'type','text'),'fontsize',7)
plotperform(tr);
savefig(tituloPerformance);
plottrainstate(tr);
savefig(tituloTrainState);
save(tituloRede, 'net');
%close all force

