function [Sounddata] = soundRec(time)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Fs = 8000;
%[d,Fs] = audioread('music.wav'); % Loading of input signal
recording = audiorecorder;
disp('Recording....');
recordblocking(recording,time); %Recording sound for 10 seconds 
Sounddata =  getaudiodata(recording);
end
