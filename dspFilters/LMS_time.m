%% Initialization
clear all;
close all;
clc;
%% Recording the the signal from mic

Fs = 8000;
recording = audiorecorder;
disp('Recording....');
%recording sound for 5 seconds 
recordblocking(recording,5); 
%obtaining the numerical array of the recorded sound
tic
data =  getaudiodata(recording); 
disp('Playing uncorrupted sound');
%plays the uncorrupted sound
sound(data,Fs) 

pause(6);
%% normalization of the signal
data = data / rms(data, 1);                
% length of the signal
sg_len = length(data);                   
t=(1:sg_len)';
%% creating White Gaussian Noise

reference_signal = wgn(sg_len,1,8); % white gaussian noise with the length of the input signal

%% designing digital filter
 % maximum no.of delay elements
order = 4;      
% designing a FIR filter for adjusting weigts
fir_fil = fir1(order, 0.6);              
% filtering the reference signal
u = filter(fir_fil, 1, reference_signal);       
%% adding noise to the recorded signal

% signal to be filtered
noise_added_signal = data + u;  

soundsc(noise_added_signal, Fs)

pause(6);
%% LMS ALgorithm for calculating weights
mu = 0.0003;     
%length of noised_signal signal
n = length(noise_added_signal);  
%initializing vectors
w = zeros(order,1);      
E = zeros(1,sg_len);
for k = order:n
 U = u(k-(order-1):k);
 % preliminary output signal 
 y = U'*w;                
 % error
 E(k) = noise_added_signal(k)-y; 
 % calculating LMS fiter weights
 w = w + mu*E(k)*U;       
end
toc;

%% PLAY FILTERED SIGNAL(OUTPUT)
soundsc(E,Fs)
pause(6);
e_lms = E;
save('C:\Users\Megha Veerendra\Desktop\DSP_Lab\Project\data_lms.mat','d','e_lms','noised_signal');
