clc;
savedir = 'E:\SDR'; % SDR image saving directory
mkdir(savedir);
file_path = 'E:\HDR_exr'; % HDR image directory
files = dir(fullfile(file_path,'*.exr')); % read all the files in the package
image_num = length(files); % total number of images
ii = 1;
for j = 1:image_num
   image_name = ['W' int2str(j) '.exr'];
   image = exrread(fullfile(file_path,image_name));
   image_crop = imcrop(image,[704 284 511 511]); % crop image to 512*512
   image_Y = RGB2Lum(image_crop); %luminance of cropped hdr image
   sum = 0;
   for i = 1:512
       for k = 1:512
           sum = sum + log(max(double(image_Y(i,k)),realmin));   
       end
   end
   G = exp(sum*(1/(512*512))); %geometric mean of image
   delta_t = 0.18/G;
   X = delta_t * image_Y; 
   Lum = double(X)./(1+double(X));
   imgOut=zeros(size(image_crop));
   for i=1:3
       imgOut(:,:,i) = image_crop(:,:,i) .* (Lum ./ image_Y) ;
   end
   % save images
   filename = [sprintf('%03d',ii) '.png'];
   fullname = fullfile(savedir,filename);
   imwrite(imgOut,fullname);
   ii = ii+1;
end 

