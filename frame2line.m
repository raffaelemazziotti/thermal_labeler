function line = frame2line(frame,id)
% line = frame2line(frame,id)
flatframe = frame(:)';
line = [num2str(id),';', strrep(num2str(flatframe),'  ',';') ];