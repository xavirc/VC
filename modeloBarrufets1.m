%% Usamos los histogramas de RGB como features

% Ruta a las imágenes modelo y no modelo

addpath('C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\barrufets\positiu');
addpath('C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\barrufets\negatiu');

carpeta_negativas = 'C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\barrufets\negatiu'; % Imágenes de Bob Esponja
carpeta_modelo = 'C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\barrufets\positiu'; % Imágenes sin Bob Esponja

% Obtener archivos
archivos_modelo = dir(fullfile(carpeta_modelo, '*.jpg'));
archivos_negativos = dir(fullfile(carpeta_negativas, '*.jpg'));

% Inicializar matriz de características y etiquetas
features2 = [];
labels2 = [];

% calcular el histograma del modelo 
numBins = 32;
%numBins = 64;


% Procesar imágenes modelo (clase 1)
for i = 1:length(archivos_modelo)
    im_scene = imread(archivos_modelo(i).name);
    
    hist_R_scene = imhist(im_scene(:,:,1), numBins) / sum(imhist(im_scene(:,:,1), numBins));
    hist_G_scene = imhist(im_scene(:,:,2), numBins) / sum(imhist(im_scene(:,:,2), numBins));
    hist_B_scene = imhist(im_scene(:,:,3), numBins) / sum(imhist(im_scene(:,:,3), numBins));

    feature_vector = [hist_R_scene; hist_G_scene; hist_B_scene];
    % utilizar los histogramas en vez de la distancia

    % Guardar en la matriz de características
    features2 = [features2; feature_vector'];
    labels2 = [labels2; 1]; % Etiqueta 1 (Bob Esponja)
end

% Procesar imágenes negativas (clase 0)
for i = 1:length(archivos_negativos)
    %im_scene = imread(archivos_negativos(i).name);
    im_scene = imread(fullfile(archivos_negativos(i).folder, archivos_negativos(i).name));

    hist_R_scene = imhist(im_scene(:,:,1), numBins) / sum(imhist(im_scene(:,:,1), numBins));
    hist_G_scene = imhist(im_scene(:,:,2), numBins) / sum(imhist(im_scene(:,:,2), numBins));
    hist_B_scene = imhist(im_scene(:,:,3), numBins) / sum(imhist(im_scene(:,:,3), numBins));


    feature_vector = [hist_R_scene; hist_G_scene; hist_B_scene];

    % Guardar en la matriz de características
    features2 = [features2; feature_vector'];
    labels2 = [labels2; 0]; % Etiqueta 0 (No Bob Esponja)
end

%% entrenar clasificador

% Dividir los datos en entrenamiento y prueba
cv = cvpartition(labels2, 'HoldOut', 0.3); % 70% entrenamiento, 30% prueba
trainIdx = training(cv); % Índices de entrenamiento
testIdx = test(cv); % Índices de prueba

X_train_2 = features2(trainIdx, :); % Características de entrenamiento
y_train_2 = labels2(trainIdx); % Etiquetas de entrenamiento
X_test_2 = features2(testIdx, :); % Características de prueba
y_test_2 = labels2(testIdx); % Etiquetas de prueba


k_values = [1, 3, 5, 7, 9];
for k = k_values
    Mdl = fitcknn(X_train_2, y_train_2, 'NumNeighbors', k);
    y_pred = predict(Mdl, X_test_2);
    accuracy = sum(y_pred == y_test_2) / length(y_test_2);
    fprintf('k = %d, Precisión: %.2f\n', k, accuracy);
end



