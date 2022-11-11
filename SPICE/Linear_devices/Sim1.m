function Sim1(x, w, ysp, cbsize, rw, nbits, ref)

cbsize=str2num(cbsize);
rw=str2num(rw);
nbits=str2num(nbits);
NoRefColns=str2num(ref);

Vin=load(x)
W=load(w);
if NoRefColns>=1
    Ysp=Ref_SPICE(Vin,W,cbsize,rw,nbits);
else
    Ysp=Double_SPICE(Vin,W,cbsize,rw,nbits);
csvwrite(ysp,full(Ysp));
end