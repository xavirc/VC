modelo_path = 'modelo.jpg';
% Ruta a las imágenes modelo y no modelo
carpeta_negativas = 'C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\BobEsponja\positiu'; % Contiene imágenes de Bob Esponja
carpeta_modelo = 'C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\BobEsponja\negatiu'; % Contiene imágenes sin Bob Esponja


% Obtener archivos
archivos_modelo = dir(fullfile(carpeta_modelo, '*.jpg'));
archivos_negativos = dir(fullfile(carpeta_negativas, '*.jpg'));

% Inicializar matriz de características y etiquetas
features = [];
labels = [];

% Procesar imágenes modelo (clase 1)
for i = 1:length(archivos_modelo)
    feature_vector = extractHistogramDistanceWithSIFT(modelo_path, archivos_modelo(i).name);

    % Guardar en la matriz de características
    features = [features; feature_vector];
    labels = [labels; 1]; % Etiqueta 1 (Bob Esponja)
end

%Procesar imágenes negativas (clase 0)
% for i = 1:length(archivos_negativos)
%     feature_vector = extractHistogramDistanceWithSIFT(modelo_path, archivos_negativos(i).name);
% 
%     % Guardar en la matriz de características
%     features = [features; feature_vector];
%     labels = [labels; 0]; % Etiqueta 0 (No Bob Esponja)
% end


features

