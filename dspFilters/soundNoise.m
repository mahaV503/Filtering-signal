
%%
function [NoiseSoundD,m]=soundNoise(d)

d = d / rms(d, 1);                % Normalization of the signal
m = length(d);                   % length of the signal
%% Creating Noise(White Gaussian Noise)
reference_signal = wgn(m,1,8); % white gaussian noise with the length of the input signal
%% Designing digital filter
FIR_fil = fir1(12, 0.6);              % Designing a FIR filter for adjusting weigts
u = filter(FIR_fil, 1, reference_signal);        % Filtering the reference signal
%% Addition of noise to the input signal
NoiseSoundD = d + u;    % signal to be filtered

end