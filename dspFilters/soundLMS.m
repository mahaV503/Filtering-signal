
%%
function [LMScancelSound]=soundLMS(m,NoiseSound)

reference_signal = wgn(m,1,8); 
FIR_fil = fir1(12, 0.6);              % Designing a FIR filter for adjusting weigts
u = filter(FIR_fil, 1, reference_signal);   
order = 12;       % order of the filter(maximum no.of delay elements)
mu = 0.0003642;     % It is constant(optimal value of mu value is calculated using experiments)
n = length(NoiseSound);     %length of noised_signal signal
LMScancelSound = zeros(order,1);      %initializing vectors
E = zeros(1,m);
for k = 12:n
 U = u(k-11:k);
 y = U'*LMScancelSound;                % preliminary output signal 
 E(k) = NoiseSound(k)-y;     % error
 LMScancelSound = LMScancelSound + mu*E(k)*U;  