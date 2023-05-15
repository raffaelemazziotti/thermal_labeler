
% TODO bug del salvataggio dell'ID a 8bit

pth = 'C:\Users\pupil\Downloads\test_1h\20230509_102104-ROI_2-ROI_2\img.csv';

tf = ThermalFile(pth);
%
ax = axes();
%ax1 = subplot(2,1,1,ax);

range = [0,255];
smoothValue = 0.02*diff(range).^2;


inc = 0;
while ishandle(ax)
    if tf.hasNext()
        frame = tf.next();
        frame(frame==0)=mean(frame(:));
        %subplot(2,1,1);

        frame = imguidedfilter(frame,"DegreeOfSmoothing",smoothValue);
        thresh = multithresh(frame,2);

        imagesc(frame);
        
        filt = imquantize(frame,thresh);
        filt = imfill(filt);

        %to_img = cat(2,frame,filt) ;
        %subplot(2,1,2);
        filt = filt>2;
        
        
        rp = regionprops(filt,frame,{'Area','Centroid','BoundingBox', 'MeanIntensity', 'MaxIntensity' 'WeightedCentroid'});
        if length(rp)>1
            [a,i] = sort([rp.Area]);
            rp = rp(i(end));
            filt = bwareaopen(filt, rp.Area);
            
        end

        rct = rectangle('Position',[rp.WeightedCentroid(1)-5, rp.WeightedCentroid(2)-5, 10, 10],'FaceColor','none','EdgeColor','r', 'LineWidth',1,'Parent',ax);
        
        %imagesc( filt )

        title(ax,['img ',num2str(inc)])
        drawnow
        pause(.1)
        inc=inc+1;
    else
        tf.reset()
        inc=0;
    end
end
disp('file closed')
tf.close()
%%

fid = fopen(pth);
%%
frewind(fid);
infor = fgetl(fid);
infor = strsplit(infor,'#');
infor = strsplit(infor{2},';');

info = struct();
for h = 1:length(infor)
    res = strsplit(infor{h},':');
    info.(res{1}) = eval(res{2});
end
%%


tline = fgetl(fid);
rawline = str2num(tline);
line_id = rawline(1);
frame = reshape( rawline(2:end), info.ROI_y, info.ROI_x);

%%

%%

tline = fgetl(fid);
while ischar(tline)
    disp(tline)
    tline = fgetl(fid);
end

%%
fclose(fid)


