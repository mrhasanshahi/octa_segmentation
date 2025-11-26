%%%%%This algorithm calculated Vessel Density in Deep and Suprficial images 
% If Deep images is used to calculate the vessel density, so in vessel
% enhancement section parameter of "t" should be assign: t = 1.5, and for
% superficial images this parameter is t =2

%%
clc;clear all;close all

%% Get Input Images
[FileName, PathName, filterindex]=uigetfile( ...
        {'*.jpg','JPEG File (*.jpg)'; ...
    '*.*','Any Image file (*.*)'});
i = fullfile(PathName,FileName);
imRaw = (imread(i));

if size(imRaw,3)==3
    imRaw=rgb2gray(imRaw);
end
Img_ORG=imRaw;

%% Preprocessing (Histogram Equalization and Filtering)
Img_ORG1 = Img_ORG;
%Img_ORG2=Img_ORG(:,:,1);
% Img = imcrop(Img); 
Img_ORG1 = im2double(Img_ORG1);
figure,imshow(Img_ORG1)
[m,n] = size(Img_ORG1);
R= min(m,n);
% R = 364;
% Img_ORG1 = imresize(Img_ORG1,[R,R]);
figure,imshow(Img_ORG1,[])
[x,y] = ginput(1);
hold on
plot(x,y,'r*')
r1 = R/6; 
r2 = R/2;
%%
im_hom=filter_homomorphic(Img_ORG1,pi/6,1.8,0.2);
figure,imshow(im_hom,[])
mn=min(im_hom(:));
mx=max(im_hom(:));
im_hom=(im_hom-mn)/(mx-mn);
%% Mean subtract
patch = roipoly(Img_ORG1); 
M = mean(Img_ORG1(patch));
Img_ORG2 = Img_ORG1;
Img_ORG2(patch)= M;
Img_denoise = imsubtract(Img_ORG2,M);
%%
se = strel('disk',4);
im_morph = imsubtract(imadd(double(Img_denoise),double(imtophat(Img_denoise,se))),double(imbothat(Img_denoise,se))); % morphology oprator
%im_morph = imsubtract((imadd(Img_denoise,imtophat(Img_denoise,se))),imbothat(Img_denoise,se)); % morphology oprator

figure,imshow(im_morph)
%% vessel enhancement
Ip = single(im_morph);
thr = prctile(Ip(Ip(:)>0),1) * .1;
Ip(Ip<=thr) = thr;
Ip = Ip - min(Ip(:));
Ip = Ip ./ max(Ip(:)); 
t = 1.5;
V = vesselness2D(imadjust(imbilatfilt(Ip)), 0.5:0.5:t, [1;1], 1, true);
%V1 = vesselness2D(Ip, 0.5:0.5:1.5, [1;1], 1, true);

figure,imshow(V)
%figure,imshow(V1)
%%
% I1 = imbinarize(V,'adaptive','Sensitivity',0.3);
% figure,imshow(I1,[])
%I2= bwareaopen(I1,10);
%%
TH = graythresh(V);
I1= double(V>TH );
figure,imshow(I1,[])
%I2= bwareaopen(I1,5);
% figure,imshow(I2,[])

%% skeletonization
Img_BIN_1 = I1;
Img_SKL_1 = bwmorph(Img_BIN_1,'skel',15);
figure,imshow(Img_BIN_1,[])
figure,imshow(Img_SKL_1,[])
 imwrite(Img_BIN_1,[PathName ,'Img_Binary.png'])
 imwrite(Img_SKL_1,[PathName ,'Img_Skeleton.png'])

%% Drowing upper and lower circles
F1 = Img_BIN_1;
%F3 = rgb2gray(F3);
%F3 = im2bw(F)
F1 = double(F1);
x_center = x;
y_center= y;

DR = insertShape(F1,'circle',[x_center y_center r1],'LineWidth',2);
DRR = insertShape(DR,'circle',[x_center y_center r2],'LineWidth',2);
figure,imshow(DRR)
imwrite(DRR,[PathName ,'Circle_Binary.png'])

for i=1:size(F1,1)
    for j=1:size(F1,2)
        if (i-y_center)^2+(j-x_center)^2 >= r2^2
            F1(i,j)=0;
        
        end
    end
end

for i=1:size(F1,1);
    for j=1:size(F1,2);
        if (i-y_center)^2+(j-x_center)^2<= r1^2;
            F1(i,j)=0;
        
        end
    end
end
figure,imshow(F1,[])
imwrite(F1,[PathName ,'parafovead_Binary.png'])
%% %% VD calculation in whole Image (Binary Image)
k=0;
l=0;
% I1 = Img_BIN_1;
for i=1:size(Img_BIN_1,1);
    for j=1:size(Img_BIN_1,2);
        if Img_BIN_1(i,j)==0;
           k=k+1;
        end
    end
end
for i=1:size(Img_BIN_1,1);
    for j=1:size(Img_BIN_1,2);
        if Img_BIN_1(i,j)==1;
           l=l+1;
        end
    end
end

jj=(l/(l+k));% VD in whole Image
retina_VAD = jj


%% VD in Ring _Binary Image
k=0;
l=0;

for i=1:size(F1,1);
    for j=1:size(F1,2);
        if (i-y_center)^2+(j-x_center)^2 <= r2^2 && (i-y_center)^2+(j-x_center)^2>= r1^2 && F1(i,j)==0
           k=k+1;
        end
    end
end
for i=1:size(F1,1);
    for j=1:size(F1,2);
        if (i-y_center)^2+(j-x_center)^2 <=r2^2 && (i-y_center)^2+(j-x_center)^2>= r1^2 && F1(i,j)==1
           l=l+1;
        end
    end
end

jj=(l/(l+k));
parafovea_VAD=jj

%% Drowing upper and lower circles in Skeleton Image
F3 = Img_SKL_1;
%F3 = rgb2gray(F3);
%F3 = im2bw(F3)
figure,imshow(F3)
% [x,y] = ginput(1);
% y_center = x;
% x_center= y;
F3 = double(F3);
DR1 = insertShape(F3,'circle',[x_center y_center r1],'LineWidth',2);
DRR1 = insertShape(DR1,'circle',[x_center y_center r2],'LineWidth',2);
figure,imshow(DRR1)
imwrite(DRR1,[PathName ,'Circle_Skeleton.png'])

for i=1:size(F3,1);
    for j=1:size(F3,2);
        if (i-y_center)^2+(j-x_center)^2 >= r2^2;
            F3(i,j)=0;
        
        end
    end
end


for i=1:size(F3,1);
    for j=1:size(F3,2);
        if (i-y_center)^2+(j-x_center)^2<= r1^2;
            F3(i,j)=0;
        
        end
    end
end
figure,imshow(F3,[])
imwrite(F3,[PathName ,'parafovead_Skeleton.png'])
%% %% VD calculation in whole Image (Binary Image)

k=0;
l=0;

for i=1:size(Img_SKL_1,1)
    for j=1:size(Img_SKL_1,2)
        if Img_SKL_1(i,j)==0
           k=k+1;
        end
    end
end
for i=1:size(Img_SKL_1,1)
    for j=1:size(Img_SKL_1,2)
        if Img_SKL_1(i,j)==1
           l=l+1;
        end
    end
end

jj=(l/(l+k));% VD in whole Image
retina_VSD = jj


%% VD in Ring _Binary Image
k=0;
l=0;
%figure,imtool(F3)
for i=1:size(F3,1)
    for j=1:size(F3,2)
        if (i-y_center)^2+(j-x_center)^2 <= r2^2 && (i-y_center)^2+(j-x_center)^2>= r1^2 && F3(i,j)==0
           k=k+1;
        end
    end
end
for i=1:size(F3,1)
    for j=1:size(F3,2)
        if (i-y_center)^2+(j-x_center)^2 <= r2^2 && (i-y_center)^2+(j-x_center)^2>= r1^2 && F3(i,j)==1
           l=l+1;
        end
    end
end

jj=(l/(l+k));
parafovea_VSD = jj
% Save the results 
%fid = fopen([PathName +'VD_without resize.txt'],'w');
fid = fopen([PathName +'VD_with resize(364).txt'],'w');
% %fid = fopen([PathName +'VD_with resize(min).txt'],'w');


fprintf(fid,'%s','retina_VAD = ');
fprintf(fid, '%0.4f\r\n\r',retina_VAD); 
fprintf(fid,'%s','parafovea_VAD = ');
fprintf(fid, '%0.4f\r\n\r',parafovea_VAD); 
fprintf(fid,'%s', 'retina_VSD = ');
fprintf(fid, '%0.4f\r\n\r',retina_VSD); 
fprintf(fid,'%s','parafovea_VSD = ');
fprintf(fid, '%0.4f\r\n\r',parafovea_VSD); 
fclose(fid)

