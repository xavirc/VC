% Cargar imagen principal y patch de referencia

%mainImagePath = 'Smurfs4565.jpg'; %foto original
%mainImagePath = 'Smurfs5594.jpg'; % apareix pero poc
mainImagePath = 'Smurfs4230.jpg'; %foto en la que no apareix el personatge

patchReferencePath = 'patch_pitufo.png';
mainImage = double(imread(mainImagePath));
patchReference = double(imread(patchReferencePath));

% Mostrar la imagen principal y el patch de referencia

%figure, imshow(patchReference / 255), title('Patch de referencia');

% Calcular histograma 2D del patch de referencia
patchReference = patchReference ./ (patchReference(:,:,1) + patchReference(:,:,2) + patchReference(:,:,3) + 1); % Normalizar
h1 = histcounts2(patchReference(:,:,1), patchReference(:,:,2), 16); % Histograma 2D
h1 = h1 / sum(h1, 'all'); % Normalizar el histograma


% Normalizar la imagen principal
mainImage = mainImage ./ (mainImage(:,:,1) + mainImage(:,:,2) + mainImage(:,:,3) + 1);

figure, imshow(mainImage), title('Imagen principal');

% Tamaño de la ventana y paso
patchSize = 128; % Tamaño del patch
step = 10;       % Paso de la ventana

% Inicializar variables para guardar el mejor patch
bestSimilarity = -Inf; % Valor más bajo posible de similitud
bestPatch = [];        % Mejor patch encontrado
bestCoordinates = [];  % Coordenadas del mejor patch

% Desplazar la ventana por la imagen
[nRows, nCols, ~] = size(mainImage);
for y = 1:step:(nRows - patchSize + 1)
    for x = 1:step:(nCols - patchSize + 1)
        % Extraer el patch actual
        patch = mainImage(y:(y + patchSize - 1), x:(x + patchSize - 1), :);
        
        % Calcular histograma 2D
        h2 = histcounts2(patch(:,:,1), patch(:,:,2), 16);
        h2 = h2 / sum(h2, 'all'); % Normalizar el histograma

        % Calcular similitud con el patch de referencia (suma de mínimos)
        similarity = sum(min(h1, h2), 'all');

        % Actualizar el mejor patch si se encuentra uno más similar
        if similarity > bestSimilarity
            bestSimilarity = similarity;
            bestPatch = patch;
            bestCoordinates = [x, y]; % Coordenadas superiores izquierda del patch
        end
    end
end

% Mostrar el mejor patch encontrado
%figure, imshow(bestPatch), title('Mejor patch encontrado');

% Mostrar imagen principal con el mejor patch destacado
figure, imshow(mainImage), title('Imagen con mejor patch destacado');
hold on;
rectangle('Position', [bestCoordinates(1), bestCoordinates(2), patchSize, patchSize], ...
          'EdgeColor', 'g', 'LineWidth', 2); % Rectángulo verde para destacar
hold off;

bestSimilarity