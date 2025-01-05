% Script para retornar la ruta de una imagen aleatoria

% Ruta base de la carpeta TRAIN
trainDir = fullfile(pwd, 'TRAIN'); % Ajusta si TRAIN no está en el directorio actual

% Obtener las carpetas de series dentro de TRAIN
seriesFolders = dir(trainDir);
seriesFolders = seriesFolders([seriesFolders.isdir] & ~startsWith({seriesFolders.name}, '.'));

% Elegir una carpeta de serie aleatoria
randomSeriesFolder = seriesFolders(randi(length(seriesFolders))).name;

% Obtener las subcarpetas 'si' y 'no'
subfolders = {'si', 'no'};
randomSubfolder = subfolders{randi(length(subfolders))};

% Ruta completa de la subcarpeta aleatoria
targetFolder = fullfile(trainDir, randomSeriesFolder, randomSubfolder);

% Obtener las imágenes .jpg dentro de la carpeta seleccionada
imageFiles = dir(fullfile(targetFolder, '*.jpg'));

% Verificar si hay imágenes en la carpeta seleccionada
if isempty(imageFiles)
    error('No hay imágenes en la carpeta seleccionada: %s', targetFolder);
end

% Elegir una imagen aleatoria
randomImage = imageFiles(randi(length(imageFiles))).name;

% Ruta completa de la imagen aleatoria
randomImagePath = fullfile(targetFolder, randomImage);

% Mostrar la ruta de la imagen seleccionada
disp(['Ruta de la imagen seleccionada: ', randomImagePath]);
