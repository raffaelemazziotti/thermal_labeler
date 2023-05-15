classdef LabelWriter < handle
    %IMGFILE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        pth
        fid
        file_info
        header_raw
    end
    
    methods
        function obj = LabelWriter(pth, header)
            
            if nargin<2 || isempty(header)
                obj.header_raw = '#';
            else
                obj.header_raw = header;
            end           

            obj.pth = pth;
            path_full = fullfile(obj.pth, 'label.csv');
            if not(exist(path_full,'file'))
                file_is_new = true;
            else
                file_is_new = false;
            end
            obj.fid = LazyFileReader(path_full);
            
            if file_is_new
                obj.fid.writeLine(obj.header_raw);
            end

            % header
            obj.header_raw = obj.fid.readLine(1);
            infor = strsplit(obj.header_raw,'#');
            infor = strsplit(infor{2},';');

            obj.file_info = struct();
            for h = 1:length(infor)
                res = strsplit(infor{h},':');
                obj.file_info.(res{1}) = eval(res{2});
            end

        end
        
        function [frame, info] = readNext(obj)
            tline = obj.fid.getNextLine();
            [frame,info] = line2frame(tline, obj.file_info.ROI_y, obj.file_info.ROI_x);
        end

        function [frame, info] = readPrevious(obj)
            tline = obj.fid.getPreviousLine();
            [frame,info] = line2frame(tline, obj.file_info.ROI_y, obj.file_info.ROI_x);
        end
        
        function [frame, info] = readLine(obj, i)
            tline = obj.fid.readLine(i);
            [frame,info] = line2frame(tline, obj.file_info.ROI_y, obj.file_info.ROI_x);
        end

        function check = hasNext(obj)
            check = obj.fid.currentLineIndex<obj.fid.totalLines;
        end

        function check = hasPrevious(obj)
            check = obj.fid.currentLineIndex>1;
        end

        function reset(obj)
            % starts from the beginning skipping the header
            obj.fid.goToLine(2);
        end

        function addLine(obj, frame, id)
            % TODO controlla se frame2line posso farla meglio
            obj.fid.writeLine(frame2line(frame,id))
        end

        function perc = progress(obj)
            perc = obj.fid.currentLineIndex / obj.fid.totalLines;
        end
        
        function close(obj)
            obj.fid.close;
        end
        
    end
end

