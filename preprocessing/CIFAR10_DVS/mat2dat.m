function [vector_allAddr, vector_allTs]=mat2dat(CIN,s)
%
% converts a matlab 6 column matrix CIN into a .dat file (whos name is contained in string s)
% for displaying with
% jAER (http://sourceforge.net/p/jaer/wiki/Home/). You should use the
% RetinaTeresa2 or tmpdiff128, DVS128 (activating filters invert-x and invert-y)
% AEChips to display the converted file in jAER.
%
% Each line in the matrix represents a recorded event.
% The 6 columns of the input matrix mean the following:
%
% Column 1: timestamps with 1us time tick
% Columns 2-3: ignore them (they are meant for simulator AERST (see Perez-Carrasco et al, IEEE
% TPAMI, Nov. 2013)).
% Column 4: x coordinate (from 0 to 127)
% Column 5: y coordinate (from 0 to 127)
% Column 6: event polarity
%
% The events in the output .dat file will be 16bits timestamp plus 16bits representing the following:
% bit 0: polarity
% bits 1 to 7:  x coordinate (from 0 to 127)
% bits 8 to 14: y coordinate (from 0 to 127)
% bit 15: ignored, set to 0.
% This distribution of bits can be changed below.
%
% Written by Jose A. Pérez-Carrasco in 2007, while at the Sevilla
% Microelectronics Institute (IMSE-CNM, CSIC and Univ. of Sevilla, Spain).



% The following lines map x,y, and event polarity into specific bits of a
% 2-byte word for displaying with jAER's filters RetinaTeresa2 or tmpdiff128.
% You can change these bits according to the jAER filters
% you may want to use.

retinaSizeX=128; 
xmask = hex2dec ('fE'); % x are 7 bits (64 cols) ranging from bit 1-7
ymask = hex2dec ('7f00'); % y are also 7 bits from bit 8 to 14.
xshift=1; % bits to shift x to right
yshift=8; % bits to shift y to right
polmask=1; % polarity bit is LSB

x=CIN(:,4);
y2=CIN(:,5);
pol=CIN(:,6);
Ax=retinaSizeX-1-x; % x addresses
xfinal=bitshift(Ax,xshift);
yfinal=bitshift(y2,yshift);

polfinal=(1-pol)/2;
vector_allAddr=uint32(yfinal+xfinal+polfinal);
vector_allTs=uint32(CIN(:,1));

f=fopen(s,'w');

tok='#!AER-DAT';
tok2='# This is a raw AE data file - do not edit';
tok3='# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event';
tok4='# Timestamps tick is 1 us';
tok5='# created Tue Apr 29 11:36:59 CEST 2008';
v=2;
bof=ftell(f);
fprintf(f,'%s',tok);
fprintf(f,'%1.1f\r\n',v);
fprintf(f,'%s\r\n',tok2);
fprintf(f,'%s\r\n',tok3);
fprintf(f,'%s\r\n',tok4);
fprintf(f,'%s\r\n',tok5);

bof=ftell(f);
bof2=bof;
fseek(f,bof-4,'bof'); % start just after header
bof=ftell(f);
tam=length(vector_allAddr);

fwrite(f,vector_allAddr,'uint32',4,'b');

bof22=bof2;
fseek(f,bof+4,'bof'); % timestamps start 4 after bof
fwrite(f,vector_allTs,'uint32',4,'b');
fclose(f);