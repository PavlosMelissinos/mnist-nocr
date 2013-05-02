function [ imOut] = image_script( images, x, y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[imagesN, imagesY, imagesX] = size(images);
imOut = zeros(x * imagesX, y * imagesY);
counter = 0;
for xi = 1:x
    %xstart = 1 + imagesX*(xi-1);
    xend = imagesX*xi;
    xstart = xend - imagesX + 1;
    for yi = 1:y
        yend = imagesY*yi;
        ystart = yend - imagesY + 1;
        counter = counter + 1;
        Ai = reshape(images(counter,:,:),imagesX,imagesY);
        Ai = Ai';
        imOut(xstart:xend,ystart:yend) = Ai;
    end
end

end

