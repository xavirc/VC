% Cargar imágenes de entrenamiento
modelo_path = 'modelo.jpg';
% Ruta a las imágenes modelo y no modelo

addpath('C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\BobEsponja\positiu');
addpath('C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\BobEsponja\negatiu');

carpeta_negativas = 'C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\BobEsponja\negatiu'; % Imágenes de Bob Esponja
carpeta_modelo = 'C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\BobEsponja\positiu'; % Imágenes sin Bob Esponja

% Obtener archivos
archivos_modelo = dir(fullfile(carpeta_modelo, '*.jpg'));
archivos_negativos = dir(fullfile(carpeta_negativas, '*.jpg'));

% Inicializar matriz de características y etiquetas
features = [];
labels = [];

im_model = imread(modelo_path);

% calcular el histograma del modelo 
numBins = 256;
%numBins = 64;

% Histograma del modelo
hist_R_model = imhist(im_model(:,:,1), numBins) / sum(imhist(im_model(:,:,1), numBins));
hist_G_model = imhist(im_model(:,:,2), numBins) / sum(imhist(im_model(:,:,2), numBins));
hist_B_model = imhist(im_model(:,:,3), numBins) / sum(imhist(im_model(:,:,3), numBins));

% Procesar imágenes modelo (clase 1)
for i = 1:length(archivos_modelo)
    im_scene = imread(archivos_modelo(i).name);
    
    hist_R_scene = imhist(im_scene(:,:,1), numBins) / sum(imhist(im_scene(:,:,1), numBins));
    hist_G_scene = imhist(im_scene(:,:,2), numBins) / sum(imhist(im_scene(:,:,2), numBins));
    hist_B_scene = imhist(im_scene(:,:,3), numBins) / sum(imhist(im_scene(:,:,3), numBins));

    dist_R = sqrt(sum((hist_R_model - hist_R_scene).^2));
    dist_G = sqrt(sum((hist_G_model - hist_G_scene).^2));
    dist_B = sqrt(sum((hist_B_model - hist_B_scene).^2));

    feature_vector = [dist_R, dist_G, dist_B];
    % utilizar los histogramas en vez de la distancia

    % Guardar en la matriz de características
    features = [features; feature_vector];
    labels = [labels; 1]; % Etiqueta 1 (Bob Esponja)
end

% Procesar imágenes negativas (clase 0)
for i = 1:length(archivos_negativos)
    %im_scene = imread(archivos_negativos(i).name);
    im_scene = imread(fullfile(archivos_negativos(i).folder, archivos_negativos(i).name));

    hist_R_scene = imhist(im_scene(:,:,1), numBins) / sum(imhist(im_scene(:,:,1), numBins));
    hist_G_scene = imhist(im_scene(:,:,2), numBins) / sum(imhist(im_scene(:,:,2), numBins));
    hist_B_scene = imhist(im_scene(:,:,3), numBins) / sum(imhist(im_scene(:,:,3), numBins));

    dist_R = sqrt(sum((hist_R_model - hist_R_scene).^2));
    dist_G = sqrt(sum((hist_G_model - hist_G_scene).^2));
    dist_B = sqrt(sum((hist_B_model - hist_B_scene).^2));

    feature_vector = [dist_R, dist_G, dist_B];

    % Guardar en la matriz de características
    features = [features; feature_vector];
    labels = [labels; 0]; % Etiqueta 0 (No Bob Esponja)
end

%% entrenar clasificador

% Dividir los datos en entrenamiento y prueba
cv = cvpartition(labels, 'HoldOut', 0.3); % 70% entrenamiento, 30% prueba
trainIdx = training(cv); % Índices de entrenamiento
testIdx = test(cv); % Índices de prueba

X_train = features(trainIdx, :); % Características de entrenamiento
y_train = labels(trainIdx); % Etiquetas de entrenamiento
X_test = features(testIdx, :); % Características de prueba
y_test = labels(testIdx); % Etiquetas de prueba

% Entrenar el clasificador k-NN
k = 3; % Número de vecinos
Mdl = fitcknn(X_train, y_train, 'NumNeighbors', k);

% Evaluar el clasificador
y_pred = predict(Mdl, X_test);

% Calcular la precisión
accuracy = sum(y_pred == y_test) / length(y_test);
disp(['Precisión del clasificador k-NN: ', num2str(accuracy)]);

