%% calcular histogramas de todas las series

% Lista de clases y sus carpetas correspondientes
clases = {'barrufets', 'bobesponja', 'gatigos', 'gumball', 'horadeaventuras', ...
    'oliverybenji', 'padredefamilia', 'pokemon', 'southpark', 'tomyjerry'};

% Ruta base donde se encuentran las carpetas de entrenamiento
ruta_base = 'TRAIN';

% Número de bins para el histograma
numBins = 64;

% Inicializar matrices para características y etiquetas
features = [];
labels = [];

% Procesar cada clase
for idx = 1:length(clases)
    clase = clases{idx};
    carpeta_clase = fullfile(ruta_base, clase); % Ruta completa a la carpeta de la clase
    
    % Obtener lista de imágenes en la carpeta
    archivos = dir(fullfile(carpeta_clase, '*.jpg'));
    
    % Procesar cada imagen en la carpeta
    for i = 1:length(archivos)
        % Leer la imagen
        im_scene = imread(fullfile(archivos(i).folder, archivos(i).name));
        
        % Calcular histogramas normalizados para cada canal RGB
        hist_R_scene = imhist(im_scene(:,:,1), numBins) / sum(imhist(im_scene(:,:,1), numBins));
        hist_G_scene = imhist(im_scene(:,:,2), numBins) / sum(imhist(im_scene(:,:,2), numBins));
        hist_B_scene = imhist(im_scene(:,:,3), numBins) / sum(imhist(im_scene(:,:,3), numBins));
        
        % Concatenar los histogramas en un vector de características
        feature_vector = [hist_R_scene; hist_G_scene; hist_B_scene];
        
        % Guardar las características y la etiqueta correspondiente
        features = [features; feature_vector'];
        labels = [labels; idx]; % Etiqueta numérica basada en el índice de la clase
    end
end

% Mostrar un resumen de los datos cargados
fprintf('Características extraídas: %d.', size(features, 1));
fprintf(' Dimensión de cada vector de características: %d', size(features, 2));

%% dividir datos en entrenamiento y prueba

% Dividir los datos en entrenamiento y prueba
cv = cvpartition(labels, 'HoldOut', 0.3); % 70% entrenamiento, 30% prueba
trainIdx = training(cv); % Índices de entrenamiento
testIdx = test(cv); % Índices de prueba

X_train = features(trainIdx, :); % Características de entrenamiento
y_train = labels(trainIdx); % Etiquetas de entrenamiento
X_test = features(testIdx, :); % Características de prueba
y_test = labels(testIdx); % Etiquetas de prueba

% Mostrar un resumen de la división
fprintf('Conjunto de entrenamiento: %d muestras\n', size(X_train, 1));
fprintf('Conjunto de prueba: %d muestras\n', size(X_test, 1));

%% Ver desempeño en subconjunto de test

% Realizar predicciones en las imágenes de prueba
y_pred = allModel.predictFcn(X_test);

% Mostrar resultados

accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('Precisión en el conjunto de prueba: %.2f%%\n', accuracy * 100);

%% Guardar modelo

save('allModel.mat', 'allModel');
