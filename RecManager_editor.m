classdef RecManager_editor<handle
    % rewrite the thermo file excluding wrong frames (no label or 'none')
    properties
        file_img
        file_label
        file_beh

        file_img_new
        file_label_new
        file_beh_new

        header_raw
        file_info
        lastWrittenLabel_id
        totalLineFrames
        totalLineLabels

        currentLine
        img_buffer_n_elem
        img_buffer
    end

    methods
        function obj = RecManager_editor(pth)
            obj.file_img = LazyFileReader2( fullfile(pth,'img.csv') );
            obj.file_label = LazyFileReader2( fullfile(pth,'label.csv') );
            obj.file_beh = LazyFileReader2( fullfile(pth,'label_beh.csv') );

            obj.file_img_new = LazyFileReader2( fullfile(pth,'img_new.csv') );
            obj.file_label_new = LazyFileReader2( fullfile(pth,'label_new.csv') );
            obj.file_beh_new = LazyFileReader2( fullfile(pth,'label_beh_new.csv') );

            obj.lastWrittenLabel_id=-1;
            obj.img_buffer_n_elem=0;

            % header
            obj.header_raw = obj.file_img.readLine(1);
            infor = strsplit(obj.header_raw,'#');
            infor = strsplit(infor{2},';');

            obj.file_info = struct();
            for h = 1:length(infor)
                res = strsplit(infor{h},':');
                obj.file_info.(res{1}) = eval(res{2});
            end
            

            obj.file_img_new.writeLine(obj.header_raw);
            obj.file_label_new.writeLine(obj.header_raw);
            obj.file_beh_new.writeLine(obj.header_raw);

            obj.totalLineFrames = obj.file_img.totalLines-1;
            obj.totalLineLabels = obj.file_label.totalLines-1;
            obj.currentLine = 1;
        end

        function sync(obj)
            disp('not implemented')
        end

        function [img, lbl, beh, id ] = getFrameNext(obj)
            [img, lbl, beh, id] = getFrameAtIndex(obj,obj.currentLine+1);
        end

        function [img, lbl, beh, id] = getFrameAtIndex(obj,index)
            raw = obj.file_img.readLine(index);
            [img, img_id] = line2frame(raw,obj.file_info.ROI_y,obj.file_info.ROI_x);
  
            raw = obj.file_label.readLine(index);
            [lbl, lbl_id] = line2frame(raw,obj.file_info.ROI_y,obj.file_info.ROI_x);
            raw = obj.file_beh.readLine(index);
            fileds = strsplit(raw,';');
            beh =  fileds{2};
            beh_id = str2num(fileds{1});

            if (img_id - lbl_id)~=0 || (img_id-beh_id) ~= 0
                error('ID are disaligned')
            end
            id = img_id;
            obj.currentLine = index;
        end

        function res = validateIndex(obj,index)
            res = index>0 && index<obj.totalLineFrames;
        end

        function res = hasNext(obj)
            res = obj.currentLine<=obj.totalLineFrames;
        end

        function res = hasPrevious(obj)
            res = obj.currentLine>2;
        end
       

        function writeLine(obj,id,img,lbl,beh)
            obj.file_img_new.writeLine(frame2line(img,id));
            obj.file_label_new.writeLine(frame2line(lbl,id));
            obj.file_beh_new.writeLine([num2str(id),';' ,beh]);

        end

        function frac = progress(obj)
            frac = obj.currentLine/obj.totalLineFrames;
        end

        function close(obj)

            if ~isempty(obj.file_img)
                obj.file_img.close;
                obj.file_img = [];
            end

            if ~isempty(obj.file_img_new)
                obj.file_img_new.close;
                obj.file_img_new = [];
            end

            if ~isempty(obj.file_label)
                obj.file_label.close;
                obj.file_label = [];
            end

            if ~isempty(obj.file_label_new)
                obj.file_label_new.close;
                obj.file_label_new = [];
            end

            if ~isempty(obj.file_beh)
                obj.file_beh.close;
                obj.file_beh = [];
            end

            if ~isempty(obj.file_beh_new)
                obj.file_beh_new.close;
                obj.file_beh_new = [];
            end

        end
    end
end


