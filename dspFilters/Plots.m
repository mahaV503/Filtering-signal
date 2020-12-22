%Plot outputs
load('data_lms.mat');
load('data_nlms.mat');
load('data_rls.mat');
m = length(d);
t=(1:m)';
%% Input Signal characteristics
figure(1);
plot(t,d)
%% Noisy signal

plot(t,noised_signal);
title('Noised Signal');
xlabel('Time');
ylabel('Amplitude');
%% LMS plots
figure(2);

plot(t,e_lms);
title('LMS PLOT');
xlabel('Time');
ylabel('Amplitude');
%% NLMS plots
figure(3);

plot(t,e_nlms);
title('NLMS PLOT');
xlabel('Time');
ylabel('Amplitude');
%% RLS plots
figure(4);

plot(t,e_rls);
title('RLS PLOT');
xlabel('Time');
ylabel('Amplitude');
%% Comparison Plots
for i=1:length(d)
error_lms(i) = abs(d(i)-e_lms(i)); 
error_nlms(i) = abs(d(i)-e_nlms(i));
error_rls(i) = abs(d(i)-e_rls(i));
end
n_error_lms = error_lms./max(error_lms);
n_error_nlms = error_lms./max(error_nlms);
n_error_rls = error_lms./max(error_rls)
figure(5);
hold on

plot(t,n_error_lms);

plot(t,n_error_nlms);

plot(t,n_error_rls);
legend('lms','nlms','rls');
%% Snr plots
snr(d) % SNR of Input signal
snr(e_lms)  % SNR of the output signal from filter
snr(e_nlms)  % SNR of the output signal from filter
snr(e_rls)  % SNR of the output signal from filter
%% MSE calculation
lms_mse = immse(d,e_lms')
nlms_mse = immse(d,e_nlms')
rls_mse = immse(d,e_rls')