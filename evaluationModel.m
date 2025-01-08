%% Evaluación de clasificadores encadenados

% Ruta base donde se encuentran las imágenes organizadas por serie
ruta_base = 'TRAIN'; % Cambia a la ruta de tus datos

% Cargar modelos entrenados
load('modelos/seriesModel.mat', 'seriesModel'); % Modelo de series

% Crear el contenedor dinámico de modelos de personajes
characterModels = containers.Map();
load('modelos/barrufets.mat', 'barrufets');
characterModels('barrufets') = barrufets;

load('modelos/bobesponja.mat', 'bobesponja');
characterModels('bobesponja') = bobesponja;

load('modelos/gatigos.mat', 'gatigos');
characterModels('gatigos') = gatigos;

load('modelos/gumball.mat', 'Gumball');
characterModels('gumball') = Gumball;

load('modelos/horadeaventuras.mat', 'horadeaventuras');
characterModels('horadeaventuras') = horadeaventuras;

load('modelos/oliverybenji.mat', 'oliverModel');
characterModels('oliverybenji') = oliverModel;

load('modelos/padredefamilia.mat', 'petterModel');
characterModels('padredefamilia') = petterModel;

load('modelos/pokemon.mat', 'ashModel');
characterModels('pokemon') = ashModel;

load('modelos/southpark.mat', 'cartmanModel');
characterModels('southpark') = cartmanModel;

load('modelos/tomyjerry.mat', 'tomModel');
characterModels('tomyjerry') = tomModel;

bins = 64;

% Inicializar contadores para métricas
total_images = 0;
correct_series_predictions = 0;
correct_character_predictions = 0;

% Lista de clases (series)
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
        predictedClassIndex = seriesModel.predictFcn(feature_vector');
        predictedClassName = series_list{predictedClassIndex};
        total_images = total_images + 1;
        
        % Comprobar si la serie fue correctamente clasificada
        if strcmp(predictedClassName, clase_real)
            correct_series_predictions = correct_series_predictions + 1;
            
            % --- Clasificar la presencia del personaje ---
            modelo_personaje = characterModels(predictedClassName); % Seleccionar modelo de personaje correspondiente
            personaje_predicho = modelo_personaje.predictFcn(feature_vector');
            
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
        predictedClassIndex = seriesModel.predictFcn(feature_vector');
        predictedClassName = series_list{predictedClassIndex};
        total_images = total_images + 1;
        
        % Comprobar si la serie fue correctamente clasificada
        if strcmp(predictedClassName, clase_real)
            correct_series_predictions = correct_series_predictions + 1;
            
            % --- Clasificar la presencia del personaje ---
            modelo_personaje = characterModels(predictedClassName); % Seleccionar modelo de personaje correspondiente
            personaje_predicho = modelo_personaje.predictFcn(feature_vector');
            
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


%% Generar gráficos de análisis

% --- Gráfico de barras: Comparación de precisiones ---
figure;
bar([precision_series, precision_character, global_precision] * 100);
set(gca, 'XTickLabel', {'Series', 'Personajes', 'Global'});
ylabel('Precisión (%)');
title('Comparación de Precisión por Clasificador');
grid on;

% --- Gráfico de tarta: Clasificaciones correctas e incorrectas ---
correct_classifications = correct_character_predictions;
incorrect_classifications = total_images - correct_classifications;
figure;
pie([correct_classifications, incorrect_classifications], ...
    {'Correctas', 'Incorrectas'});
title('Distribución de Clasificaciones Correctas vs Incorrectas');

% --- Gráfico de barras agrupadas: Precisión por serie ---
series_correct = zeros(1, length(series_list));
series_total = zeros(1, length(series_list));

% Calcular precisiones por serie
for i = 1:length(series_list)
    serie = series_list{i};
    carpeta_si = fullfile(ruta_base, serie, 'si');
    carpeta_no = fullfile(ruta_base, serie, 'no');
    
    series_total(i) = numel(dir(fullfile(carpeta_si, '*.jpg'))) + ...
                      numel(dir(fullfile(carpeta_no, '*.jpg')));
    series_correct(i) = correct_series_predictions / length(series_list); % Suponiendo distribución uniforme
end

series_precision = (series_correct ./ series_total) * 100;

% Crear gráfico
figure;
bar(categorical(series_list), series_precision, 'FaceColor', 'flat');
xlabel('Series');
ylabel('Precisión (%)');
title('Precisión por Serie');
grid on;

