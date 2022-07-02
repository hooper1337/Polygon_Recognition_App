clc;
clear;
close all;

totalTeste = 0;
totalTotal = 0;
neuronios = 10;
camadas = [neuronios neuronios neuronios neuronios neuronios neuronios];
list = dir('Photos/start/**/*.png');

filenames = string({list.folder}) + '/'+string({list.name});
str = filenames;
i=1;
count = 0;

for st = str
    count = count +1;
end

img_res = [28 28];
poligonos = zeros(img_res(1)*img_res(2)*3,count);

for st = str
    numChar = strfind(st,".");
    S = extractBefore(st, numChar);
    numChar1 = strfind(S,"/");
    St2 = extractBefore(S, numChar1);
    numChar2 = strfind(S,"/");
    St3 = extractAfter(S, numChar2);
    fileStList = St2+'\'+St3+'.png';
    I = imread(fileStList);
    I = imresize(I,img_res);
    BinImage = imbinarize(I); size(BinImage);
    poligonos(:,i) = reshape(BinImage, 1, []);
    fileout = "Out/Binary/" + St3 + ".bin";
    fileID = fopen(fileout, 'w');
    fprintf(fileID, '%d', poligonos(:,i));
    i=i+1;
end

letrasTarget = [1 1 1 1 1 ...
    2 2 2 2 2 ...
    3 3 3 3 3 ...
    4 4 4 4 4 ...
    5 5 5 5 5 ...
    6 6 6 6 6 ...
    ];

letrasTarget = onehotencode(letrasTarget,1,'ClassNames',1:6);

net = feedforwardnet(camadas);

net.trainFcn = 'traingdx';
net.layers{end}.transferFcn = 'purelin';
net.layers{1:end-1}.transferFcn = 'tansig';
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;
net.trainParam.epochs = 1000;

%Train
[net,tr] = train(net, poligonos, letrasTarget);
%Test
out = net(poligonos);
disp(tr);
r=0;
for i=1:size(out,2)               
    [a, b] = max(out(:,i));         
    [c, d] = max(letrasTarget(:,i));  
    if b == d                       
        r = r+1;
    end
end
plotconfusion(out, letrasTarget);

TTargets = letrasTarget(tr.testInd);
out_test = (out(tr.testInd)>0.5);
accuracy = sum(out_test == TTargets)/length(tr.testInd);
fprintf('Precisao total = %f\n', accuracy*100);

