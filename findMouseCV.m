function filt = findMouseCV(frame)

    frame(frame==0)=mean(frame(:));
    %subplot(2,1,1);
    
    range = [0,255];
    smoothValue = 0.02*diff(range).^2;

    frame = imguidedfilter(frame,"DegreeOfSmoothing",smoothValue);
    thresh = multithresh(frame,2);
    
    filt = imquantize(frame,thresh);
    filt = imfill(filt);
    
    %to_img = cat(2,frame,filt) ;
    %subplot(2,1,2);
    filt = filt>2;