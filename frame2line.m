function line = frame2line(frame,id)
% line = frame2line(frame,id)

line = [num2str(id),';', sprintf('%.1f;' , frame(:))];
line = line(1:end-1);