clear all,close all,clc


cd /Users/hglover/Library/CloudStorage/Box-Box/HannahGlover/Research/data/icesat2

fid = fopen('icesat2_3873_formatlab.csv');
formatspec='%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f'; %14

T = textscan(fid,formatspec,'HeaderLines',1,'Delimiter',',');

