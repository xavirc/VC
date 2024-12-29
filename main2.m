% Main program for image classification using allModel2 (Personajes Específicos)

load('allModel2.mat', 'allModel2');  % Cargar el modelo entrenado

% Lista de personajes específicos (clases)
personajes_seleccionados = {'gran barrufet', 'gat i gos', 'Gumball', 'Finn', ...
    'Oliver', 'Bob esponja', 'Peter Griffin', 'Ash Ketchum', 'Cartman', 'Tom'};

while true
    try
        % Preguntar al usuario por el nombre de la imagen
        imageName = input('Enter image name (or type "exit" to quit): ', 's');
        
        % Verificar si el usuario quiere salir
        if strcmpi(imageName, 'exit')
            break;
        end
        
        % Leer la imagen
        img = imread(imageName);
        
        % Redimensionar la imagen al tamaño fijo (128x128)
        targetSize = [128, 128];
        img = imresize(img, targetSize);
        
        % Características de color (histogramas RGB)
        numBins = 32;
        hist_R = imhist(img(:,:,1), numBins) / numel(img(:,:,1));
        hist_G = imhist(img(:,:,2), numBins) / numel(img(:,:,2));
        hist_B = imhist(img(:,:,3), numBins) / numel(img(:,:,3));
        color_features = [hist_R; hist_G; hist_B]';
        
        % Características de forma (momentos de Hu)
        BW = imbinarize(rgb2gray(img));
        stats = regionprops(BW, 'Eccentricity', 'Extent');
        if ~isempty(stats)
            hu_features = [stats(1).Eccentricity, stats(1).Extent]; % Usar la primera región encontrada
        else
            hu_features = [0, 0]; % En caso de que no haya una región identificable
        end
        
        % Características de textura (HOG)
        grayIm = rgb2gray(img);
        HOG_features = extractHOGFeatures(grayIm, 'CellSize', [8 8]);
        
        % Concatenar todas las características
        feature_vector = [color_features, hu_features, HOG_features];
        
        % Realizar predicción usando el modelo entrenado allModel2
        predictedClassIndex = predict(allModel2, feature_vector);
        
        % Obtener el nombre del personaje predicho
        predictedCharacterName = personajes_seleccionados{predictedClassIndex};
        
        % Mostrar el resultado
        fprintf('Predicted Character: %s\n', predictedCharacterName);
        
    catch ME
        % Mostrar mensaje de error y traza de pila
        fprintf('Error: %s\n', ME.message);
        fprintf('Stack trace:\n');
        disp(ME.stack);
        
        % Salir del programa
        break;
    end
end
