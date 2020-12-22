%NLMS Algo
clc;
clear all;
%% Audio is recorded through the mic 
%sampling Frequency
Fs = 8000; 
%number of channels
nChannel = 1; 
%greater number of samples implies better sound quality
bits_per_sample = 24; 
%% recording audio recording is done using audiorecorder function
recording = audiorecorder(Fs,bits_per_sample,nChannel);
disp('Recording....');
recordblocking(recording,5); %Recording sound for 5 seconds 
tic
% obtaining the numerical array of the recorded sound
data =  getaudiodata(recording); 
disp('Playing uncorrupted sound');
% plays the uncorrupted sound
sound(data,Fs); 
pause(6);
%% normalization of the signal
data = data / rms(data, 1);                

%% corrupting Audio Signal with noise
% filter length
M = 4; 
% designing a FIR filter for adjusting weights
FIR_fil = fir1(M, 0.6);              

% noise is added to the audio input to corrupt it
noise = wgn(length(data),1,10); 
% noise is added to the uncorrupted signal. 
noise_added_signal = data + noise; 
u = filter(FIR_fil, 1, noise_added_signal);
disp('Playing corrupted sound');
sound(noise_added_signal,Fs,bits_per_sample); %Playing the corrupted sound
pause(6);

%% NLMS parameters
% Step size
mu = 0.0003; 
% bias 
epsilon = 0.1; 
w = zeros(M,1);

for i = M:length(noise_added_signal)
 U = u(i-(M-1):i);
 % difference between LMS and NLMS
 K = mu/(abs(epsilon+(data(i)^2))); 
 % preliminary output signal
 y = U'*w;         
 % error
 E(i) = data(i)-y;     
 % cal NLMS fiter weights
 w = w + K*E(i)*U;       
end

disp('Filtered Signal');
sound(E,Fs);
pause(6);
e_nlms = E;
%Saving data to a file
save('C:\Users\Megha Veerendra\Desktop\DSP_Lab\Project\data_nlms.mat','e_nlms');
toc