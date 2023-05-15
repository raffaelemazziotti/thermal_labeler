function [frame, id] = line2frame(line,size_x,size_y)
% [frame, id] = line2frame(line,size_x,size_y)

rawline = str2num(line);
id = rawline(1);
frame = reshape( rawline(2:end), size_x, size_y);
