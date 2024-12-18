function feature_vector = extractHistogramDistanceWithSIFT(im_model_path, im_scene_path)
    %% Cargar imágenes
    im_model = imread(im_model_path);
    im_model_grey = rgb2gray(im_model);

    im_scene = imread(im_scene_path);
    im_scene_grey = rgb2gray(im_scene);

    %% Detectar características SIFT y describir
    kp_model = detectSIFTFeatures(im_model_grey);
    kp_scene = detectSIFTFeatures(im_scene_grey);

    [feat_model, kp_model] = extractFeatures(im_model_grey, kp_model);
    [feat_scene, kp_scene] = extractFeatures(im_scene_grey, kp_scene);

    %% Aparear puntos clave
    pairs = matchFeatures(feat_model, feat_scene, 'MatchThreshold', 10);
    matched_kp_model = kp_model(pairs(:,1), :);
    matched_kp_scene = kp_scene(pairs(:,2), :);


 
    %% Calcular transformación geométrica
    try
        %[tform, inliers] = estimateGeometricTransform2D(matched_kp_model, matched_kp_scene, 'affine', 'MaxNumTrials', 2000, 'MaxDistance', 3);
        [tform, inliers] = estimateGeometricTransform2D(matched_kp_model, matched_kp_scene, 'affine');
    catch
        % Si ocurre un error, asignar NaN y continuar
        % warning('Error en la estimación de la transformación geométrica. Se devuelven NaN.');
        feature_vector = [NaN, NaN, NaN, NaN, NaN, NaN, 0];
        return;
    end

    inliers_kp_model = matched_kp_model(inliers, :);
    inliers_kp_scene = matched_kp_scene(inliers, :);

    %% Recortar la región detectada en la escena
    tformInv = invert(tform);
    roi_scene = imwarp(im_scene, tformInv, 'OutputView', imref2d(size(im_model_grey)));

    %% Calcular histogramas normalizados
    numBins = 256;
    % Histograma del modelo
    hist_R_model = imhist(im_model(:,:,1), numBins) / sum(imhist(im_model(:,:,1), numBins));
    hist_G_model = imhist(im_model(:,:,2), numBins) / sum(imhist(im_model(:,:,2), numBins));
    hist_B_model = imhist(im_model(:,:,3), numBins) / sum(imhist(im_model(:,:,3), numBins));

    % Histograma de la región detectada
    hist_R_scene = imhist(roi_scene(:,:,1), numBins) / sum(imhist(roi_scene(:,:,1), numBins));
    hist_G_scene = imhist(roi_scene(:,:,2), numBins) / sum(imhist(roi_scene(:,:,2), numBins));
    hist_B_scene = imhist(roi_scene(:,:,3), numBins) / sum(imhist(roi_scene(:,:,3), numBins));

    %% Calcular distancias y correlaciones
    % Distancias euclidianas
    dist_R = sqrt(sum((hist_R_model - hist_R_scene).^2));
    dist_G = sqrt(sum((hist_G_model - hist_G_scene).^2));
    dist_B = sqrt(sum((hist_B_model - hist_B_scene).^2));

    % Correlaciones
    corrR = corr2(hist_R_model, hist_R_scene);
    corrG = corr2(hist_G_model, hist_G_scene);
    corrB = corr2(hist_B_model, hist_B_scene);

    %% Crear vector de características
    feature_vector = [dist_R, dist_G, dist_B, corrR, corrG, corrB, length(inliers_kp_scene)];

end