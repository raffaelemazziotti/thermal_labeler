classdef RecManager<handle
    %RECMANAGER Summary of this class goes here
    %   Detailed explanation goes here

    properties
        file_img
        file_label
        img_total
        img_current_index
        img_buffer
        img_buffer_n_elem
    end

    methods
        function obj = RecManager(pth)
            obj.file_img = ImgFile(pth);
            obj.file_label = LabelWriter(pth, obj.file_img.header_raw);
            obj.img_total = obj.file_img.fid.totalLines-1;
            obj.sync();
            obj.img_buffer_n_elem=0;
        end

        function sync(obj)
            obj.file_img.fid.goToLine(obj.file_label.fid.totalLines);
            obj.file_label.fid.goToLine(obj.file_label.fid.totalLines);
            obj.img_current_index = obj.file_img.fid.currentLineIndex;
        end

        function [img , lbl, img_id,is_editable] = getNext(obj)
            if obj.file_img.hasNext
                [img, img_id] = obj.file_img.readNext;

                if obj.file_label.hasNext
                    [lbl, lbl_id] = obj.file_label.readNext;
                    is_editable = 0;
                else
                    if ~isempty(obj.img_buffer)
                        dove = ismember([obj.img_buffer(:).id], img_id);
                    else
                        dove = 0;
                    end

                    if any(dove)
                        lbl_id = obj.img_buffer(find(dove)).id;
                        lbl = obj.img_buffer(find(dove)).lbl;
                        is_editable = 1;
                    else
                        lbl = findMouseCV(img);
                        lbl_id = img_id;
                        obj.img_buffer_n_elem = obj.img_buffer_n_elem + 1;
                        obj.img_buffer(obj.img_buffer_n_elem).lbl = lbl;
                        obj.img_buffer(obj.img_buffer_n_elem).id = lbl_id;
                        is_editable = 1;
                    end

                end

                obj.img_current_index = obj.file_img.fid.currentLineIndex;

                if img_id - lbl_id ~= 0
                    error('ID are disaligned')
                end
            else
                img=[];
                lbl=[];
                img_id=[];
                is_editable = 0;
            end
        end

        function [img , lbl, img_id, is_editable] = getPrevious(obj)
            if obj.file_img.hasPrevious
                [img, img_id] = obj.file_img.readPrevious;

                if ~isempty(obj.img_buffer)
                    dove = ismember([obj.img_buffer(:).id], img_id);
                    is_editable = 1;
                else
                    dove = 0;
                    is_editable = 0;
                end

                if any(dove)
                    pos = find(dove);

                    lbl_id = obj.img_buffer(pos).id;
                    lbl = obj.img_buffer(pos).lbl;
                    
                    
                else
                    if obj.file_label.hasPrevious
                        [lbl, lbl_id] = obj.file_label.readLine(obj.img_current_index-1);
                        
                    else
                        img=[];
                        lbl=[];
                        img_id=[];
                        
                        return
                    end
                end

                obj.img_current_index = obj.file_img.fid.currentLineIndex;

                if img_id - lbl_id ~= 0
                    error('ID are disaligned')
                end

            else
                img=[];
                lbl=[];
                img_id=[];
                is_editable = 0;
            end
        end

        function editLabel(obj,lbl,id)
            dove = ismember([obj.img_buffer(:).id], id);
            obj.img_buffer(find(dove)).lbl = lbl;
        end

        function writeBuffer(obj)
            for b=1:length(obj.img_buffer)
                obj.file_label.addLine(obj.img_buffer(b).lbl, obj.img_buffer(b).id);
            end
            obj.img_buffer = [];
            obj.img_buffer_n_elem=0;
            obj.sync()
        end

        function check = hasNext(obj)
            check = obj.file_img.hasNext;
        end

        function check = hasPrevious(obj)
            check = obj.file_img.hasPrevious;
        end

        function frac = progress(obj)
            frac = obj.file_img.progress;
        end

        function close(obj)
            obj.file_img.close
            obj.file_label.close
        end

    end
end

