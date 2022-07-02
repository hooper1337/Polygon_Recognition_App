clc;
clear;
close all;

load ('Out\Redes\C_TrainTest_TrainPolig.mat', 'net');
list = dir('Photos/train/**/*.png');
filenames = string({list.folder}) + '/'+string({list.name});
str = filenames;
i=1;
count=0;

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

    fileID = fopen (fileout, 'w'); 
    fprintf(fileID, '%d', poligonos(:,i)); 

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
out = sim(net, poligonos);

r=0;
for i=1:size(out,2)               
    [a, b] = max(out(:,i));      
    [c, d] = max(letrasTarget(:,i));
    if b == d                      
      r = r+1;
    end
end

plotconfusion(letrasTarget, out);
accuracy = r/size(out,2);

fprintf('Precisao total = %f\n', accuracy*100)