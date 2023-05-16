classdef ImgFile < handle
    %IMGFILE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        pth
        fid
        file_info
        header_raw
    end
    
    methods
        function obj = ImgFile(pth, filename)
            
            if nargin < 2 || isempty(filename)
                filename = 'img.csv';
            end

            obj.pth = pth;
            obj.fid = LazyFileReader(fullfile(obj.pth, filename));

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

        function check = hasNext(obj)
            check = obj.fid.currentLineIndex<obj.fid.totalLines;
        end

        function check = hasPrevious(obj)
            check = obj.fid.currentLineIndex>2;
        end

        function reset(obj)
            % starts from the beginning skipping the header
            obj.fid.goToLine(2);
        end

        function perc = progress(obj)
            perc = obj.fid.currentLineIndex / obj.fid.totalLines;
        end
        
        function close(obj)
            obj.fid.close;
        end
        
    end
end

