%Implementation of RLS Adaptive Filter
clc;
clear all;
%% Audio is input through the mic 
Fs = 8000; %Sampling Frequency
nChannel = 1; %number of channels
bits_per_sample = 24; %Greater number of samples implies better sound quality
%% Recording audio recording is done using audiorecorder function
recording = audiorecorder(Fs,bits_per_sample,nChannel);
disp('Recording....');
recordblocking(recording,5); %Recording sound for 10 seconds 
tic

d =  getaudiodata(recording); %obtaining the numerical array of the recorded sound
disp('Playing uncorrupted sound');
sound(d,Fs); %Plays the uncorrupted sound
pause(6);
%% Noise is added to the uncorrupted signal. 
noise = wgn(length(d),1,10); % A gussian noise of power 10dbw is added to the audio input inorder to corrupt it
x = d + noise; % Uncorrupted sound + noise
disp('Playing corrupted sound');
sound(x,Fs,bits_per_sample) %Playing the corrupted sound
pause(6);
%% RLS Filter Implementation
%% Parameter List
% d - Reference signal
% x - Noissy signal
% lamda - Forgetting factor
% delta - Large positive quantity
% Rn - Covariance Matrix
%%
M = 4; % Length of FIR Filter
w = zeros(M,1); %Initialize filter weights to zero
delta = 100; %Large positive constant
Rn = eye(M)*delta; %Initialization of the covariance matrix
y = zeros(length(d),1); %output signal
error = zeros(length(d),1); % error signal
x_ = zeros(length(d),1);
lambda = 0.999;
F = rand(length(d),1);
for i = 1 : length( d )
    disp(i)
f( 1 : i ) = flipud( x( 1 : i ) );     
 if length( f ) < M 
 f( i + 1 : M, 1 ) = 0;
 elseif length( f ) > M            %Implementation of the rls Algorithm
 f = f( 1 : M );
 end 
K = ( Rn * f ) ./ ( lambda + (f' * Rn * f ));
e( i ) = d( i ) - (f' * w);
w= w + (K * e( i ));
Rn     = ( lambda^-1 * Rn ) - ( lambda^-1 * K * f' * Rn );
error(i)=(e(i)-d(i)).^2;
end
%Filtered sound
disp('Playing Filtered sound');
sound(e,Fs);
pause(6);
e_rls = e;
save('C:\Users\Megha Veerendra\Desktop\DSP_Lab\Project\data_rls.mat','e_rls');
toc