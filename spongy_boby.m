%% ini 
im_obj=imread('modelo.jpg');
%figure,imshow(im_obj),title('imatge model');
im_obj=rgb2gray(im_obj);

im_esc=rgb2gray(imread('SPONGE_BOB7842.jpg'));
figure,imshow(im_esc),title('objecte a reconeixer');

%% detecció i descripcio

kp_obj=detectSIFTFeatures(im_obj);
kp_esc=detectSIFTFeatures(im_esc);

figure,imshow(im_obj);
title('50 keypoints principals')
hold on
plot(selectStrongest(kp_obj, 50));


figure,imshow(im_esc);
hold on
plot(kp_esc);

[feat_obj, kp_obj] = extractFeatures(im_obj, kp_obj);
[feat_esc, kp_esc] = extractFeatures(im_esc, kp_esc);

%% aparellament

pairs=matchFeatures(feat_obj, feat_esc, 'MatchThreshold',10);

matched_kp_obj=kp_obj(pairs(:,1),:);
matched_kp_esc=kp_esc(pairs(:,2),:);

figure;
showMatchedFeatures(im_obj, im_esc, matched_kp_obj, matched_kp_esc, 'montage');
title('aparellament putatius');

%% matching

[tform, inliers] = estimateGeometricTransform2D(matched_kp_obj, matched_kp_esc, 'affine');
% [tform, inliers] = estimateGeometricTransform2D(matched_kp_obj, matched_kp_esc, 'similarity', 'MaxNumTrials', 2000, 'Confidence', 99.9);
% [tform, inliers] = estimateGeometricTransform2D(matched_kp_obj, matched_kp_esc, 'projective', 'MaxNumTrials', 2000, 'Confidence', 99.9);
inliers_kp_obj=matched_kp_obj(inliers, :);
inliers_kp_esc=matched_kp_esc(inliers, :);

[miday midax]=size(im_obj);
box_obj=[1,1;midax,1;midax,miday;1,miday;1,1];
figure,imshow(im_obj);
hold on;
line(box_obj(:,1),box_obj(:,2), 'color', 'y');
title('bounding box');

box_esc = transformPointsForward(tform, box_obj);

figure,imshow(im_esc);
hold on;
line(box_esc(:,1),box_esc(:,2), 'color', 'y');
title('matching');


