function out=filter_homomorphic(im,sigma,gHigh,gLow)
logIm = log(double(im)+1);

sIm = size(im);
sPre = ceil(sIm/2);
sPost = floor(sIm/2);

logIm_pad=padarray(logIm,sPre,1,'pre');
logIm_pad=padarray(logIm_pad,sPost,1,'post');

kx = linspace(0,2*pi,2*sIm(1)+1);
kx = kx(1:end-1); 
kx(sIm(1)+2:end)=kx(sIm(1)+2:end)-2*pi;

ky = linspace(0,2*pi,2*sIm(2)+1);
ky = ky(1:end-1);
ky(sIm(2)+2:end)=ky(sIm(2)+2:end)-2*pi;
[Kx,Ky]=ndgrid(kx,ky);
H = gLow+(gHigh-gLow)*(1-exp(-0.5*(Kx.^2+ Ky.^2)/sigma^2) );

out=ifft(fft(logIm_pad).*H);
out=out(1+sPre(1):end-sPre(1),1+sPre(2):end-sPre(2));
out=exp(real(out)-1);

