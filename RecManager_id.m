classdef RecManager_id<handle

    properties
        file_img
        file_label
        file_beh
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
        function obj = RecManager_id(pth)
            obj.file_img = LazyFileReader2( fullfile(pth,'img.csv') );
            obj.file_label = LazyFileReader2( fullfile(pth,'label.csv') );
            obj.file_beh = LazyFileReader2( fullfile(pth,'label_beh.csv') );
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
            

            if obj.file_label.totalLines==0
                obj.file_label.writeLine(obj.header_raw);
            end

            if obj.file_beh.totalLines==0
                obj.file_beh.writeLine(obj.header_raw);
            end

            obj.totalLineFrames = obj.file_img.totalLines-1;
            obj.totalLineLabels = obj.file_label.totalLines-1;

            if obj.file_label.totalLines==1
                obj.currentLine = 2;
            else
                obj.currentLine = obj.totalLineLabels;
            end

            
            obj.sync();
            
        end

        function sync(obj)
            obj.file_label.countLines();
            obj.file_beh.countLines();
            
            if obj.file_label.totalLines>1
                ids = [];
                for l=2:obj.file_label.totalLines
                    content = obj.file_label.readLine(l);
                    content = strsplit(content,';');
                    ids(l-1) = str2num(content{1});
                end
                if diff(ids)<=0
                    error('indexes appear not properly alligned')
                end
                obj.lastWrittenLabel_id=max(ids);
            end
        end
    
        function [img, lbl, beh, id, is_editable] = getFrameCurrent(obj)
            [img, lbl, beh, id, is_editable] = getFrameAtIndex(obj,obj.currentLine);
        end

        function [img, lbl, beh, id, is_editable] = getFrameNext(obj)
            [img, lbl, beh, id, is_editable] = getFrameAtIndex(obj,obj.currentLine+1);
        end

        function [img, lbl, beh, id, is_editable] = getFramePrevious(obj)
            [img, lbl, beh, id, is_editable] = getFrameAtIndex(obj,obj.currentLine-1);
        end

        function [img, lbl, beh, id, is_editable] = getFrameAtIndex(obj,index)
            raw = obj.file_img.readLine(index);
            [img, img_id] = line2frame(raw,obj.file_info.ROI_y,obj.file_info.ROI_x);
            is_editable = 0;
            if img_id > obj.lastWrittenLabel_id
                is_editable = 1;
                if ~isempty(obj.img_buffer)
                    dove = ismember([obj.img_buffer(:).id], img_id);
                else
                    dove = 0;
                end
                
                if any(dove)
                    lbl = obj.img_buffer(dove).lbl;
                    beh = obj.img_buffer(dove).beh;
                    lbl_id = obj.img_buffer(dove).id;
                    beh_id = obj.img_buffer(dove).id;
                else

                    lbl = findMouseCV(img);
                    lbl_id = img_id;
                    beh_id = img_id;
                    beh = 'NONE';

                    obj.img_buffer_n_elem = obj.img_buffer_n_elem + 1;
                    obj.img_buffer(obj.img_buffer_n_elem).lbl = lbl;
                    obj.img_buffer(obj.img_buffer_n_elem).beh = beh;
                    obj.img_buffer(obj.img_buffer_n_elem).id = lbl_id;
                end
            else
                raw = obj.file_label.readLine(index);
                [lbl, lbl_id] = line2frame(raw,obj.file_info.ROI_y,obj.file_info.ROI_x);
                raw = obj.file_beh.readLine(index);
                fileds = strsplit(raw,';');
                beh =  fileds{2};
                beh_id = str2num(fileds{1});
            end
            
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
        
        function editLabel(obj,lbl,beh,id)
            dove = ismember([obj.img_buffer(:).id], id);
            obj.img_buffer(find(dove)).lbl = lbl;
            obj.img_buffer(find(dove)).beh = beh;
        end

        function writeBuffer(obj)
            if ~isempty(obj.img_buffer)
                ids = [obj.img_buffer.id];
                
                % controla se quello che verrà scritto già c'è
                if min(ids)<=obj.lastWrittenLabel_id
                    error('ID overlap error!')
                end

                % riordino il buffer
                [t,order] = sort(ids);
                obj.img_buffer = obj.img_buffer(order);

                for b=1:length(obj.img_buffer)
                    obj.file_label.writeLine(frame2line(obj.img_buffer(b).lbl, obj.img_buffer(b).id));
                    obj.file_beh.writeLine([num2str(obj.img_buffer(b).id),';' ,obj.img_buffer(b).beh]);
                end
                obj.img_buffer = [];
                obj.img_buffer_n_elem=0;
                obj.sync();
            end
        end

        function frac = progress(obj)
            frac = obj.currentLine/obj.totalLineFrames;
        end

        function close(obj)

            if ~isempty(obj.file_img)
                obj.file_img.close;
                obj.file_img = [];
            end

            if ~isempty(obj.file_label)
                obj.file_label.close;
                obj.file_label = [];
            end

            if ~isempty(obj.file_beh)
                obj.file_beh.close;
                obj.file_beh = [];
            end

        end
    end
end

