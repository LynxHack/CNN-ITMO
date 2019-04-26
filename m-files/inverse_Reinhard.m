image = imread('0820.png');
ImgIn=(double(ImgIn)./255).^2.2;
I = RGB2Lum(ImgIn);
X = double(I)./(double(I)-1);
sum = 0;
for i = 1:size(image,1)
    for k = 1:size(image,2)
        sum = sum + log(max(double(X(i,k)),realmin));   
    end
end
G_X = exp(sum*(1/(size(image,1)*size(image,2)))); %geometric mean of image
P = size(image,1)*size(image,2);
PB1 = size(find(I==0),1);
PB2 = size(find(I~=0),1);
a = 0.18;
G_E = exp(P*log(G_X)/PB1-PB2*log(a)/PB1);
E = G_E*X/a;
E(isnan(E))=realmin;
E(E>=2^32) = 2^32;
imgOut=zeros(size(ImgIn));
for i=1:3
    imgOut(:,:,i) = double(ImgIn(:,:,i)) .* ( E./ double(I)) ;
end
