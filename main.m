% Main program for image classification

load('allModel.mat', 'allModel');

% Lista de clases y sus carpetas correspondientes
clases = {'barrufets', 'bobesponja', 'gatigos', 'gumball', 'horadeaventuras', ...
    'oliverybenji', 'padredefamilia', 'pokemon', 'southpark', 'tomyjerry'};

while true
    try
        % Ask user for image name
        imageName = input('Enter image name (or type "exit" to quit): ', 's');
        
        % Check if user wants to exit
        if strcmpi(imageName, 'exit')
            break;
        end
        
        % Read the image
        img = imread(imageName);
        
        % Extract RGB histogram features
        features = extractRGBHistogram(img);

        
        % Reshape features to match the expected format (1 x N)
        features = features(:)';  % Convert to row vector
        
        % Make prediction using the trained model's predictFcn
        predictedClassIndex = allModel.predictFcn(features)
        
        
        predictedClassName = clases{predictedClassIndex}


        

        
    catch ME
        % Display error message and stack trace
        fprintf('Error: %s\n', ME.message);
        fprintf('Stack trace:\n');
        disp(ME.stack);
        
        % Exit the program
        break;
    end
end

function features = extractRGBHistogram(img)
    % Asegúrate de que la imagen es RGB
    if size(img, 3) ~= 3
        img = repmat(img, [1, 1, 3]);
    end
    
    bins = 64;

    % Calcula los histogramas RGB
    rHist = imhist(img(:, :, 1), bins);
    gHist = imhist(img(:, :, 2), bins);
    bHist = imhist(img(:, :, 3), bins);
    
    % Normaliza los histogramas para que sumen 1
    rHist = rHist / sum(rHist);
    gHist = gHist / sum(gHist);
    bHist = bHist / sum(bHist);
    
    % Combina los histogramas en un único vector de características
    features = [rHist; gHist; bHist];
end

% Smurfs6522.jpg