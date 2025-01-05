%% Evaluación de clasificadores encadenados

% Ruta base donde se encuentran las imágenes organizadas por serie
ruta_base = 'ruta_a_las_carpetas';

% Cargar modelos entrenados
...

bins = 64;

% Inicializar contadores para métricas
total_images = 0;
correct_series_predictions = 0;
correct_character_predictions = 0;

% Lista de clases y sus carpetas correspondientes
series_list = {'barrufets', 'bobesponja', 'gatigos', 'gumball', 'horadeaventuras', ...
    'oliverybenji', 'padredefamilia', 'pokemon', 'southpark', 'tomyjerry'};


% Obtener lista de carpetas de series
carpetas_series = dir(ruta_base);
carpetas_series = carpetas_series([carpetas_series.isdir] & ~startsWith({carpetas_series.name}, '.'));

% Recorrer cada carpeta de serie
for i = 1:length(carpetas_series)
    clase_real = carpetas_series(i).name;
    carpeta_si = fullfile(ruta_base, clase_real, 'si');
    carpeta_no = fullfile(ruta_base, clase_real, 'no');
    
    % --- Procesar imágenes de la carpeta "si" (con el personaje) ---
    archivos_si = dir(fullfile(carpeta_si, '*.jpg'));
    for j = 1:length(archivos_si)
        im_scene = imread(fullfile(archivos_si(j).folder, archivos_si(j).name));
        feature_vector = extractRGBHistogram(im_scene, bins); % Extraer características (64 bins)
        
        % --- Clasificar la serie ---
        serie_predicha = predict(modeloSerie, feature_vector');
        total_images = total_images + 1;
        
        % Comprobar si la serie fue correctamente clasificada
        if strcmp(series_list{serie_predicha}, clase_real)
            correct_series_predictions = correct_series_predictions + 1;
            
            % --- Clasificar la presencia del personaje ---
            modelo_personaje = modelosPersonajes{serie_predicha}; % Seleccionar modelo de personaje correspondiente
            personaje_predicho = predict(modelo_personaje, feature_vector');
            
            % Etiqueta real: 1 (contiene al personaje)
            if personaje_predicho == 1
                correct_character_predictions = correct_character_predictions + 1;
            end
        end
    end
    
    % --- Procesar imágenes de la carpeta "no" (sin el personaje) ---
    archivos_no = dir(fullfile(carpeta_no, '*.jpg'));
    for j = 1:length(archivos_no)
        im_scene = imread(fullfile(archivos_no(j).folder, archivos_no(j).name));
        feature_vector = extractRGBHistogram(im_scene, bins); 
        
        % --- Clasificar la serie ---
        serie_predicha = predict(modeloSerie, feature_vector');
        total_images = total_images + 1;
        
        % Comprobar si la serie fue correctamente clasificada
        if strcmp(series_list{serie_predicha}, clase_real)
            correct_series_predictions = correct_series_predictions + 1;
            
            % --- Clasificar la presencia del personaje ---
            modelo_personaje = modelosPersonajes{serie_predicha}; % Seleccionar modelo de personaje correspondiente
            personaje_predicho = predict(modelo_personaje, feature_vector');
            
            % Etiqueta real: 0 (no contiene al personaje)
            if personaje_predicho == 0
                correct_character_predictions = correct_character_predictions + 1;
            end
        end
    end
end

%% Cálculo de métricas

% Precisión del clasificador de series
precision_series = correct_series_predictions / total_images;

% Precisión del clasificador de personajes (solo considerando las imágenes con serie correctamente clasificada)
precision_character = correct_character_predictions / correct_series_predictions;

% Precisión global del sistema (serie + personaje correctos)
global_precision = correct_character_predictions / total_images;

% Mostrar resultados
fprintf('Precisión del clasificador de series: %.2f%%\n', precision_series * 100);
fprintf('Precisión del clasificador de personajes: %.2f%%\n', precision_character * 100);
fprintf('Precisión global del sistema: %.2f%%\n', global_precision * 100);

%% Función auxiliar: Extracción de características
function features = extractRGBHistogram(img, bins)
    % Asegurarse de que la imagen sea RGB
    if size(img, 3) ~= 3
        img = repmat(img, [1, 1, 3]);
    end
    
    % Calcular histogramas RGB
    rHist = imhist(img(:, :, 1), bins);
    gHist = imhist(img(:, :, 2), bins);
    bHist = imhist(img(:, :, 3), bins);
    
    % Normalizar histogramas
    rHist = rHist / sum(rHist);
    gHist = gHist / sum(gHist);
    bHist = bHist / sum(bHist);
    
    % Combinar histogramas en un vector
    features = [rHist; gHist; bHist];
end
