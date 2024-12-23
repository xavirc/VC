% Configuración
carpeta_series = 'C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN'; % Ruta de la carpeta principal donde están las series
carpeta_negatiu = 'C:\Users\pauma\OneDrive\Escritorio\VC\TRAIN\barrufets\negatiu'; % Ruta de la carpeta negatiu
n = 13; % Número de imágenes aleatorias a seleccionar por serie

% Obtener las carpetas de series
carpetas = dir(carpeta_series);
carpetas = carpetas([carpetas.isdir]); % Filtrar sólo carpetas
carpetas = carpetas(~ismember({carpetas.name}, {'.', '..'})); % Eliminar '.' y '..'

% Procesar cada serie
for i = 1:length(carpetas)
    nombre_serie = carpetas(i).name;
    
    if (nombre_serie ~= "barrufets") 
        ruta_serie = fullfile(carpeta_series, nombre_serie);

        % Obtener archivos de imágenes en la carpeta actual
        archivos = dir(fullfile(ruta_serie, '*.*'));
        archivos = archivos(~[archivos.isdir]); % Filtrar sólo archivos
        extensiones_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}; % Extensiones válidas
        archivos_validos = archivos(contains(lower({archivos.name}), extensiones_validas));
        
        % Seleccionar n imágenes aleatorias
        num_imagenes = length(archivos);
        if num_imagenes < n
            fprintf('Advertencia: Solo hay %d imágenes en %s. Usando todas.\n', num_imagenes, nombre_serie);
            indices = 1:num_imagenes;
        else
            indices = randperm(num_imagenes, n);
        end
        
        % Copiar las imágenes seleccionadas a la carpeta negatiu
        for j = 1:length(indices)
            archivo_seleccionado = archivos(indices(j)).name;
            ruta_origen = fullfile(ruta_serie, archivo_seleccionado);
            ruta_destino = fullfile(carpeta_negatiu, archivo_seleccionado);
            copyfile(ruta_origen, ruta_destino);
        end
        
        fprintf('%d imágenes de %s añadidas a %s.\n', length(indices), nombre_serie, carpeta_negatiu); 
    end
end
fprintf('Proceso completado.\n');
