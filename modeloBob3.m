%% Usamos el histograma de nivel de gris como features

% Cargar imágenes de entrenamiento
modelo_path = 'modelo.jpg';

addpath('C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\BobEsponja\positiu');
addpath('C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\BobEsponja\negatiu');

carpeta_negativas = 'C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\BobEsponja\negatiu'; % Imágenes sin Bob Esponja
carpeta_modelo = 'C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\BobEsponja\positiu'; % Imágenes de Bob Esponja

% Obtener archivos
archivos_modelo = dir(fullfile(carpeta_modelo, '*.jpg'));
archivos_negativos = dir(fullfile(carpeta_negativas, '*.jpg'));

% Inicializar matriz de características y etiquetas
features3 = [];
labels3 = [];

% Número de bins para el histograma
numBins = 32;

% Procesar imágenes modelo (clase 1)
for i = 1:length(archivos_modelo)
    im_scene = imread(fullfile(archivos_modelo(i).folder, archivos_modelo(i).name));
    
    % Convertir a escala de grises
    gray_img = rgb2gray(im_scene);
    
    % Calcular histograma normalizado
    hist_gray = imhist(gray_img, numBins) / numel(gray_img);
    
    % Guardar en la matriz de características
    features3 = [features3; hist_gray']; % Transponer para fila
    labels3 = [labels3; 1]; % Etiqueta 1 (Bob Esponja)
end

% Procesar imágenes negativas (clase 0)
for i = 1:length(archivos_negativos)
    im_scene = imread(fullfile(archivos_negativos(i).folder, archivos_negativos(i).name));
    
    % Convertir a escala de grises
    gray_img = rgb2gray(im_scene);
    
    % Calcular histograma normalizado
    hist_gray = imhist(gray_img, numBins) / numel(gray_img);
    
    % Guardar en la matriz de características
    features3 = [features3; hist_gray']; % Transponer para fila
    labels3 = [labels3; 0]; % Etiqueta 0 (No Bob Esponja)
end

%% Entrenar clasificador


% Dividir los datos en entrenamiento y prueba
cv = cvpartition(labels3, 'HoldOut', 0.3); % 70% entrenamiento, 30% prueba
trainIdx = training(cv); % Índices de entrenamiento
testIdx = test(cv); % Índices de prueba

X_train = features3(trainIdx, :); % Características de entrenamiento
y_train = labels3(trainIdx); % Etiquetas de entrenamiento
X_test = features3(testIdx, :); % Características de prueba
y_test = labels3(testIdx); % Etiquetas de prueba

% Probar diferentes valores de k para k-NN
k_values = [1, 3, 5, 7, 9];
for k = k_values
    Mdl = fitcknn(X_train, y_train, 'NumNeighbors', k);
    y_pred = predict(Mdl, X_test);
    accuracy = sum(y_pred == y_test) / length(y_test);
    fprintf('k = %d, Precisión: %.2f\n', k, accuracy);
end


Ensemble_bagged_trees.predictFcn

disp(Ensemble_bagged_trees.ClassificationEnsemble);