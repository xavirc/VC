%% Cargar imágenes
im_obj = imread('Smurf.jpg');
figure, imshow(im_obj), title('Imagen modelo (Pitufo)');
im_obj = rgb2gray(im_obj);

im_esc = rgb2gray(imread('Smurfs4906.jpg'));
figure, imshow(im_esc), title('Imagen a reconocer');

%% Detectar características SIFT
kp_obj = detectSIFTFeatures(im_obj);
kp_esc = detectSIFTFeatures(im_esc);

% Mostrar keypoints principales
% figure, imshow(im_obj);
% title('Keypoints principales (modelo)');
% hold on;
% plot(selectStrongest(kp_obj, 100));

% figure, imshow(im_esc);
% title('Keypoints principales (escena)');
% hold on;
% plot(selectStrongest(kp_esc, 500));


% Extraer descriptores de características
[feat_obj, kp_obj] = extractFeatures(im_obj, kp_obj);
[feat_esc, kp_esc] = extractFeatures(im_esc, kp_esc);

%% Coincidencias entre características
pairs = matchFeatures(feat_obj, feat_esc, 'MatchThreshold', 80); % Ajustar MatchThreshold

% Mostrar coincidencias putativas
matched_kp_obj = kp_obj(pairs(:, 1), :);
matched_kp_esc = kp_esc(pairs(:, 2), :);

figure;
showMatchedFeatures(im_obj, im_esc, matched_kp_obj, matched_kp_esc, 'montage');
title('Coincidencias putativas');

%% Estimar transformación geométrica
if size(matched_kp_obj, 1) >= 3
    % Probar con "similarity" primero
    [tform, inliers] = estimateGeometricTransform2D(matched_kp_obj, matched_kp_esc, 'similarity');

    % Si falla, probar con "projective"
    if isempty(inliers) || size(inliers, 1) < 3
        disp('Usando modelo projective debido a falta de inliers.');
        [tform, inliers] = estimateGeometricTransform2D(matched_kp_obj, matched_kp_esc, 'projective');
    end

    % Visualizar los inliers
    inliers_kp_obj = matched_kp_obj(inliers, :);
    inliers_kp_esc = matched_kp_esc(inliers, :);

    figure;
    showMatchedFeatures(im_obj, im_esc, inliers_kp_obj, inliers_kp_esc, 'montage');
    title('Coincidencias después de filtrar inliers');

    % Dibujar bounding box en el modelo
    [miday, midax] = size(im_obj);
    box_obj = [1, 1; midax, 1; midax, miday; 1, miday; 1, 1];

    figure, imshow(im_obj);
    hold on;
    line(box_obj(:, 1), box_obj(:, 2), 'Color', 'y', 'LineWidth', 2);
    title('Bounding box en el modelo');

    % Proyectar el bounding box a la escena
    box_esc = transformPointsForward(tform, box_obj);

    % Verificar si el bounding box es válido
    if any(isnan(box_esc), 'all')
        disp('Error: Bounding box proyectado contiene NaN.');
    else
        figure, imshow(im_esc);
        hold on;
        line(box_esc(:, 1), box_esc(:, 2), 'Color', 'y', 'LineWidth', 2);
        title('Bounding box en la escena (corregido)');
    end
else
    disp('No hay suficientes coincidencias para calcular la transformación.');
end
