%% Entrenar allModel2 para Personajes Específicos

% Lista de carpetas y personajes asociados
clases = {'barrufets', 'bobesponja', 'gatigos', 'gumball', 'horadeaventuras', ...
    'oliverybenji', 'padredefamilia', 'pokemon', 'southpark', 'tomyjerry'};
personajes_seleccionados = {'gran barrufet', 'gat i gos', 'Gumball', 'Finn', ...
    'Oliver', 'Bob esponja', 'Peter Griffin', 'Ash Ketchum', 'Cartman', 'Tom'};

% Ruta base donde se encuentran las carpetas de entrenamiento
ruta_base = 'TRAIN';

% Número de bins para histogramas de color
numBins = 32;

% Tamaño fijo para redimensionar imágenes
targetSize = [128, 128]; % Ajustar según la resolución deseada

% Inicializar matrices para características y etiquetas
features = [];
labels = [];

% Procesar cada clase y personaje seleccionado
for idx = 1:length(personajes_seleccionados)
    clase = clases{idx}; % Carpeta correspondiente al personaje
    personaje = personajes_seleccionados{idx};
    carpeta_personaje = fullfile(ruta_base, clase); % Ruta completa a la carpeta de la clase
    
    % Obtener lista de imágenes en la carpeta
    archivos = dir(fullfile(carpeta_personaje, '*.jpg'));
    
    % Procesar cada imagen en la carpeta
    for i = 1:length(archivos)
        % Leer la imagen
        im_scene = imread(fullfile(archivos(i).folder, archivos(i).name));
        
        % Redimensionar la imagen al tamaño fijo
        im_scene = imresize(im_scene, targetSize);
        
        % Características de color (histogramas RGB)
        hist_R_scene = imhist(im_scene(:,:,1), numBins) / numel(im_scene(:,:,1));
        hist_G_scene = imhist(im_scene(:,:,2), numBins) / numel(im_scene(:,:,2));
        hist_B_scene = imhist(im_scene(:,:,3), numBins) / numel(im_scene(:,:,3));
        color_features = [hist_R_scene; hist_G_scene; hist_B_scene]';
        
        % Características de forma (momentos de Hu)
        BW = imbinarize(rgb2gray(im_scene));
        stats = regionprops(BW, 'Eccentricity', 'Extent');
        if ~isempty(stats)
            hu_features = [stats(1).Eccentricity, stats(1).Extent]; % Usar la primera región encontrada
        else
            hu_features = [0, 0]; % En caso de que no haya una región identificable
        end
        
        % Características de textura (HOG)
        grayIm = rgb2gray(im_scene);
        HOG_features = extractHOGFeatures(grayIm, 'CellSize', [8 8]);
        
        % Concatenar todas las características
        feature_vector = [color_features, hu_features, HOG_features];
        
        % Verificar la consistencia de las dimensiones
        if isempty(features)
            % Inicializar matriz con el tamaño correcto
            features = zeros(0, length(feature_vector));
        elseif size(feature_vector, 2) ~= size(features, 2)
            % Ajustar el tamaño si es inconsistente
            warning('Dimensiones inconsistentes en HOG; ajustando.');
            feature_vector = feature_vector(1:size(features, 2));
        end
        
        % Guardar las características y la etiqueta correspondiente
        features = [features; feature_vector];
        labels = [labels; idx]; % Etiqueta numérica basada en el índice del personaje
    end
end

% Mostrar un resumen de los datos cargados
fprintf('Características extraídas: %d.\n', size(features, 1));
fprintf('Dimensión de cada vector de características: %d.\n', size(features, 2));

%% Dividir datos en entrenamiento y prueba

% Dividir los datos en entrenamiento y prueba
cv = cvpartition(labels, 'HoldOut', 0.3); % 70% entrenamiento, 30% prueba
trainIdx = training(cv); % Índices de entrenamiento
testIdx = test(cv); % Índices de prueba

X_train = features(trainIdx, :); % Características de entrenamiento
y_train = labels(trainIdx); % Etiquetas de entrenamiento
X_test = features(testIdx, :); % Características de prueba
y_test = labels(testIdx); % Etiquetas de prueba

% Mostrar un resumen de la división
fprintf('Conjunto de entrenamiento: %d muestras.\n', size(X_train, 1));
fprintf('Conjunto de prueba: %d muestras.\n', size(X_test, 1));

%% Entrenar un Clasificador Multiclase

% Entrenar un modelo multinomial utilizando SVM
allModel2 = fitcecoc(X_train, y_train, 'Coding', 'onevsall', 'Learners', 'Linear');

% Guardar el modelo entrenado
save('allModel2.mat', 'allModel2');

%% Evaluar el Modelo en el Conjunto de Prueba

% Realizar predicciones en las imágenes de prueba
y_pred = predict(allModel2, X_test);

% Calcular precisión
accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('Precisión en el conjunto de prueba: %.2f%%\n', accuracy * 100);

%% Guardar modelo

save('allModel2.mat', 'allModel2');