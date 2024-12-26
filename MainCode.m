%% Identificación de Personajes en Múltiples Imágenes

% Configuración inicial
addpath('TRAIN/BobEsponja/positiu', 'TRAIN/BobEsponja/negatiu');
addpath('TRAIN/barrufets/positiu', 'TRAIN/barrufets/negatiu');

% Función para entrenar un clasificador basado en histogramas RGB
function [Mdl, accuracy] = entrenarClasificador(carpeta_positivas, carpeta_negativas, numBins)
    % Cargar imágenes positivas
    archivos_positivos = dir(fullfile(carpeta_positivas, '*.jpg'));
    features = [];
    labels = [];
    
    % Procesar imágenes positivas (Clase 1)
    for i = 1:length(archivos_positivos)
        im = imread(fullfile(archivos_positivos(i).folder, archivos_positivos(i).name));
        hist_R = imhist(im(:,:,1), numBins) / numel(im(:,:,1));
        hist_G = imhist(im(:,:,2), numBins) / numel(im(:,:,2));
        hist_B = imhist(im(:,:,3), numBins) / numel(im(:,:,3));
        feature_vector = [hist_R; hist_G; hist_B]';
        features = [features; feature_vector];
        labels = [labels; 1]; % Clase positiva
    end
    
    % Cargar imágenes negativas
    archivos_negativos = dir(fullfile(carpeta_negativas, '*.jpg'));
    for i = 1:length(archivos_negativos)
        im = imread(fullfile(archivos_negativos(i).folder, archivos_negativos(i).name));
        hist_R = imhist(im(:,:,1), numBins) / numel(im(:,:,1));
        hist_G = imhist(im(:,:,2), numBins) / numel(im(:,:,2));
        hist_B = imhist(im(:,:,3), numBins) / numel(im(:,:,3));
        feature_vector = [hist_R; hist_G; hist_B]';
        features = [features; feature_vector];
        labels = [labels; 0]; % Clase negativa
    end
    
    % Dividir datos en entrenamiento y prueba
    cv = cvpartition(labels, 'HoldOut', 0.3);
    X_train = features(training(cv), :);
    y_train = labels(training(cv));
    X_test = features(test(cv), :);
    y_test = labels(test(cv));
    
    % Entrenar clasificador KNN con k óptimo
    k = 3; % Número de vecinos, puede ajustarse
    Mdl = fitcknn(X_train, y_train, 'NumNeighbors', k);
    y_pred = predict(Mdl, X_test);
    accuracy = sum(y_pred == y_test) / length(y_test);
end

% Entrenar clasificadores para cada personaje
numBins = 32;
[MdlBob, accBob] = entrenarClasificador('TRAIN/BobEsponja/positiu', 'TRAIN/BobEsponja/negatiu', numBins);
[MdlPitufos, accPitufos] = entrenarClasificador('TRAIN/barrufets/positiu', 'TRAIN/barrufets/negatiu', numBins);

fprintf('Precisión Bob Esponja: %.2f\n', accBob);
fprintf('Precisión Pitufos: %.2f\n', accPitufos);

% Función para clasificar múltiples imágenes
function clasificarVariasImagenes(MdlBob, MdlPitufos, carpeta_imagenes, numBins)
    archivos = dir(fullfile(carpeta_imagenes, '*.jpg'));
    resultados = cell(length(archivos), 2); % Guardar nombres y resultados

    for i = 1:length(archivos)
        % Leer imagen
        imagen_path = fullfile(archivos(i).folder, archivos(i).name);
        im = imread(imagen_path);

        % Calcular histogramas RGB
        hist_R = imhist(im(:,:,1), numBins) / numel(im(:,:,1));
        hist_G = imhist(im(:,:,2), numBins) / numel(im(:,:,2));
        hist_B = imhist(im(:,:,3), numBins) / numel(im(:,:,3));
        feature_vector = [hist_R; hist_G; hist_B]';

        % Predecir con ambos modelos
        pred_Bob = predict(MdlBob, feature_vector);
        pred_Pitufos = predict(MdlPitufos, feature_vector);

        % Determinar resultado
        if pred_Bob == 1 && pred_Pitufos == 0
            clasificacion = 'Bob Esponja';
        elseif pred_Bob == 0 && pred_Pitufos == 1
            clasificacion = 'Papa Pitufo';
        else
            clasificacion = 'Ninguno';
        end

        % Guardar resultados
        resultados{i, 1} = archivos(i).name;
        resultados{i, 2} = clasificacion;
        fprintf('Imagen: %s, Clasificación: %s\n', archivos(i).name, clasificacion);
    end

    % Guardar resultados en un archivo
    resultados_table = cell2table(resultados, 'VariableNames', {'Imagen', 'Clasificacion'});
    writetable(resultados_table, 'resultados_clasificacion.csv');
    fprintf('Resultados guardados en resultados_clasificacion.csv\n');
end

% Clasificar todas las imágenes de una carpeta
carpeta_imagenes = 'TRAIN/barrufets/positiu'; % Reemplazar con la ruta real
clasificarVariasImagenes(MdlBob, MdlPitufos, carpeta_imagenes, numBins);
