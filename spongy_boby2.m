%% ini 
im_model=imread('modelo.jpg');
%figure,imshow(im_obj),title('imatge model');
im_obj_grey=rgb2gray(im_model);

im_esc=imread('SPONGE_BOB7842.jpg');
im_esc_gray=rgb2gray(im_esc);

figure,imshow(im_esc_gray),title('objecte a reconeixer');

%% detecció i descripcio

kp_obj=detectSIFTFeatures(im_obj_grey);
kp_esc=detectSIFTFeatures(im_esc_gray);

[feat_obj, kp_obj] = extractFeatures(im_obj_grey, kp_obj);
[feat_esc, kp_esc] = extractFeatures(im_esc_gray, kp_esc);

%% aparellament

pairs=matchFeatures(feat_obj, feat_esc, 'MatchThreshold',10);

matched_kp_obj=kp_obj(pairs(:,1),:);
matched_kp_esc=kp_esc(pairs(:,2),:);
%% matching

[tform, inliers] = estimateGeometricTransform2D(matched_kp_obj, matched_kp_esc, 'affine');
% [tform, inliers] = estimateGeometricTransform2D(matched_kp_obj, matched_kp_esc, 'similarity', 'MaxNumTrials', 2000, 'Confidence', 99.9);
% [tform, inliers] = estimateGeometricTransform2D(matched_kp_obj, matched_kp_esc, 'projective', 'MaxNumTrials', 2000, 'Confidence', 99.9);
inliers_kp_obj=matched_kp_obj(inliers, :);
inliers_kp_esc=matched_kp_esc(inliers, :);

[miday midax]=size(im_obj_grey);
box_obj=[1,1;midax,1;midax,miday;1,miday;1,1];
box_esc = transformPointsForward(tform, box_obj);

%% Recortar la región detectada en la escena
% Crear la transformación inversa para recortar la región
tformInv = invert(tform);
roi_scene = imwarp(im_esc, tformInv, 'OutputView', imref2d(size(im_obj_grey)));

% Mostrar la región recortada
figure, imshow(roi_scene);
title('Región detectada recortada en la escena');

%% Calcular el histogramas
% Histograma del modelo
numBins = 256; % Número de bins

hist_R_model = imhist(im_model(:,:,1), numBins);
hist_G_model = imhist(im_model(:,:,2), numBins);
hist_B_model = imhist(im_model(:,:,3), numBins);

% Histograma de la región detectada
hist_R_scene = imhist(roi_scene(:,:,1), numBins);
hist_G_scene = imhist(roi_scene(:,:,2), numBins);
hist_B_scene = imhist(roi_scene(:,:,3), numBins);

% Normalizar histogramas
hist_R_model = hist_R_model / sum(hist_R_model);
hist_G_model = hist_G_model / sum(hist_G_model);
hist_B_model = hist_B_model / sum(hist_B_model);

hist_R_scene = hist_R_scene / sum(hist_R_scene);
hist_G_scene = hist_G_scene / sum(hist_G_scene);
hist_B_scene = hist_B_scene / sum(hist_B_scene);

% Mostrar histograma de la región detectada
% figure;
% bar(hist_scene);
% title('Histograma de la región detectada');

%% Comparar histogramas 

%Distancia euclidea
dist_R = sqrt(sum((hist_R_model - hist_R_scene).^2));
dist_G = sqrt(sum((hist_G_model - hist_G_scene).^2));
dist_B = sqrt(sum((hist_B_model - hist_B_scene).^2));

% fprintf('Distancia en canal R: %f\n', dist_R);
% fprintf('Distancia en canal G: %f\n', dist_G);
% fprintf('Distancia en canal B: %f\n', dist_B);

% Correlación
corrR = corr2(hist_R_model, hist_R_scene);
corrG = corr2(hist_G_model, hist_G_scene);
corrB = corr2(hist_B_model, hist_B_scene);

% fprintf('Correlación en canal R: %f\n', corrR);
% fprintf('Correlación en canal G: %f\n', corrG);
% fprintf('Correlación en canal B: %f\n', corrB);