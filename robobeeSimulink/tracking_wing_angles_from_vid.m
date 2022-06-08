%% Making a list of file locations

clear all; close all;
freqs = [20 40 60 80 100 120];
for i= 1:6
    for j= 1:4
        
        filenames{i,j} = sprintf("F:\\Videos\\oneWingRobobeeHighSpeed\\Dec3_SynchAsynchTrans\\AVI\\%dHz_Test%d.avi",freqs(i),j);

    end
end

%%

% mm = VideoReader('/Volumes/External_SSD_3/Dropbox/20Hz_Test1_compressed.avi');
% filename = 'F:\Videos\oneWingRobobeeHighSpeed\Dec3_SynchAsynchTrans\AVI\20Hz_Test2.avi';
for f = 2:length(filenames(:))
        mm = VideoReader(filenames{f})
    
     %% load vids
    
    JJ = 30;
    
    vid = [];
    
    for kk=1:JJ
        
        tmp = read(mm, kk);
        %     tmp = sum(tmp(:,:,:),3);
        %     tmp = rgb2gray(tmp);
        
        vid(:,:,kk) = tmp;
        
        
        kk
    end
    %% Find wing base location
    clf
    imagesc(vid(:,:,10));
    
    % Enter the coordinates of the wing base
    X_cent = 220;
    Y_cent = 122;
    
    hold on
    scatter(X_cent,Y_cent,'r')
    
    %% Make Mask
    [X, Y] = meshgrid(1:size(vid,2), 1:size(vid,1));
    
    X = X - X_cent;
    Y = Y - Y_cent;
    
    [theta, rho] = cart2pol(X, Y);
    
    % Make an annulus about the wing base to isolate a section of wing near the
    % tip
    R1 = 40;
    R2 = 20;
    mask = (rho<R1 & rho>R2 & Y > 0);
    
    imagesc(double(mask).*vid(:,:,1));
    
    %% Compute the background
    
    % Choose 30 random frames
    indices = randi(mm.NumFrames, [30, 1]);
    bkg = [];
    
    % Read and add them to a new array
    for kk=1:length(indices)
        
        tmp = read(mm, indices(kk));
        bkg(:,:,kk) = tmp;
        
        clf
        imagesc(bkg(:,:,kk))
        drawnow
        kk
    end
    
    % The background is the median of all the frames. Subtracting this from the
    % video will isolate just the wing
    vid_bkg = median(bkg,3);
    
    clf
    imagesc(vid_bkg)
    hold on
    scatter(X_cent,Y_cent)
    
    %% Plot just the part of the wing inside the annulus
%     
%     thresh = 0.80;
%     
%     for kk=4000:5:mm.NumFrames
%         
%         tmp = read(mm, kk);
%         
%         %     imagesc(double(tmp)./vid_bkg.*double(mask));
%         thresh_temp = (double(tmp)./vid_bkg < thresh).*mask;
%         
%         imagesc(thresh_temp)
%         hold on
%         scatter(X_cent,Y_cent)
%         
%         drawnow
%         
%     end
    %%
    thresh = 0.80;
    angle = [];
    clf
    axis([0,10000,-inf,inf])
    clear ang_tan
    for kk=1:1:mm.NumFrames
        
        % Get frame
        tmp = read(mm, kk);
        
        % remove background, apply threshold, apply mask
        thresh_tmp = (double(tmp)./vid_bkg < thresh).*mask;
        
        thresh_im   = thresh_tmp(:,:);
        
        % This leaves a small region of the wing as the only nonzero values
        % Now we find the centroid of the image
        %     b = bwconncomp(thresh_im);
        %     s = regionprops(b);
        %     [m,ind] = max([s.Area]);
        s = regionprops(thresh_im,'Centroid');
        centroid = s.Centroid;
        
        % Get the angle of the wing between the two points
        ang_tan(kk) = atan2(centroid(1)-X_cent,centroid(2)-Y_cent);
        
        % %     Plotting
        %     clf
        %     imagesc(thresh_im);
        %     hold on;
        %     scatter(X_cent,Y_cent,200,'.r')
        %     scatter(centroid(1),centroid(2),200,'.g')
        %     line([X_cent centroid(1)],[Y_cent centroid(2)]);
        %     drawnow
        
        %     clf
        %     pc = pcolor(X,Y,thresh_im);
        %     pc.EdgeAlpha = 0;
        %     hold on;
        %     scatter(0,0,200,'.r')
        %     scatter(centroid(1)-X_cent,centroid(2)-Y_cent,200,'.g')
        %     line([0 centroid(1)-X_cent],[0 centroid(2)-Y_cent]);
        % %     axis([min(X) max(X) min(Y) max(Y)])
        %     drawnow
        
        kk
        %     plot(ang_tan)
        %     axis([0,10000,-inf,inf])
        %     drawnow
    end
    
    t = linspace(-2,2,mm.NumFrames);
    
    clf
    hold on
    plot(t,detrend(ang_tan))
    xline(0)
    
    %%
    fstr    = filenames{f};
    fchar   = fstr{:};
    fsave   = strcat(fchar(61:end-4),'_vid.mat')
    save(fsave,'t','ang_tan')
    
end    

    %% load vids
    
    JJ = 30;
    
    vid = [];
    
    for kk=1:JJ
        
        tmp = read(mm, kk);
        %     tmp = sum(tmp(:,:,:),3);
        %     tmp = rgb2gray(tmp);
        
        vid(:,:,kk) = tmp;
        
        
        kk
    end
    
    
    %% Find wing base location
    clf
    imagesc(vid(:,:,1));
    
    % Enter the coordinates of the wing base
    X_cent = 220;
    Y_cent = 122;
    
    hold on
    scatter(X_cent,Y_cent,'r')
    %% Make Mask
    [X, Y] = meshgrid(1:size(vid,2), 1:size(vid,1));
    
    X = X - X_cent;
    Y = Y - Y_cent;
    
    [theta, rho] = cart2pol(X, Y);
    
    % Make an annulus about the wing base to isolate a section of wing near the
    % tip
    R1 = 40;
    R2 = 20;
    mask = (rho<R1 & rho>R2 & Y > 0);
    
    imagesc(double(mask).*vid(:,:,1));
    
    
    %% Compute the background
    
    % Choose 30 random frames
    indices = randi(mm.NumFrames, [30, 1]);
    bkg = [];
    
    % Read and add them to a new array
    for kk=1:length(indices)
        
        tmp = read(mm, indices(kk));
        bkg(:,:,kk) = tmp;
        
        clf
        imagesc(bkg(:,:,kk))
        drawnow
        kk
    end
    
    % The background is the median of all the frames. Subtracting this from the
    % video will isolate just the wing
    vid_bkg = median(bkg,3);
    
    clf
    imagesc(vid_bkg.*double(mask))
    hold on
    scatter(X_cent,Y_cent)
    
    
    %% Plot just the part of the wing inside the annulus
    
    thresh = 0.80;
    
    for kk=4000:5:mm.NumFrames
        
        tmp = read(mm, kk);
        
        %     imagesc(double(tmp)./vid_bkg.*double(mask));
        thresh_temp = (double(tmp)./vid_bkg < thresh).*mask;
        
        imagesc(thresh_temp)
        hold on
        scatter(X_cent,Y_cent)
        
        drawnow
        
    end
    
    %% Find the angle of the wing at each frame
    
    thresh = 0.80;
    angle = [];
    clf
    axis([0,10000,-inf,inf])
    clear ang_tan
    for kk=1:1:mm.NumFrames
        
        % Get frame
        tmp = read(mm, kk);
        
        % remove background, apply threshold, apply mask
        thresh_tmp = (double(tmp)./vid_bkg < thresh).*mask;
        
        thresh_im   = thresh_tmp(:,:);
        
        % This leaves a small region of the wing as the only nonzero values
        % Now we find the centroid of the image
        %     b = bwconncomp(thresh_im);
        %     s = regionprops(b);
        %     [m,ind] = max([s.Area]);
        s = regionprops(thresh_im,'Centroid');
        centroid = s.Centroid;
        
        % Get the angle of the wing between the two points
        ang_tan(kk) = atan2(centroid(1)-X_cent,centroid(2)-Y_cent);
        
        % %     Plotting
        %     clf
        %     imagesc(thresh_im);
        %     hold on;
        %     scatter(X_cent,Y_cent,200,'.r')
        %     scatter(centroid(1),centroid(2),200,'.g')
        %     line([X_cent centroid(1)],[Y_cent centroid(2)]);
        %     drawnow
        
        %     clf
        %     pc = pcolor(X,Y,thresh_im);
        %     pc.EdgeAlpha = 0;
        %     hold on;
        %     scatter(0,0,200,'.r')
        %     scatter(centroid(1)-X_cent,centroid(2)-Y_cent,200,'.g')
        %     line([0 centroid(1)-X_cent],[0 centroid(2)-Y_cent]);
        % %     axis([min(X) max(X) min(Y) max(Y)])
        %     drawnow
        
        kk
        %     plot(ang_tan)
        %     axis([0,10000,-inf,inf])
        %     drawnow
    end
    
    t = linspace(-2,2,mm.NumFrames);
    
    clf
    hold on
    plot(t,detrend(ang_tan))
    xline(0)
    
    %%
    save('20Hz_time_angle.mat','t','ang_tan')


%%

sd = load('ScopeData_20Hz_t1.mat');

philTime = sd.ScopeData.time;
philDisp = sd.ScopeData.signals(5).values;

timeLags = 7.001; %7:0.00005:7.002

for i = 1:length(timeLags)
    
    timeShift = timeLags(i);
    
    dispInterp  = interp1(philTime-timeShift,philDisp,t);
    
%     
%     figure(1)
%     hold on
%     plot(t,detrend(ang_tan))
%     plot(philTime-7.001,detrend(philDisp))
%     hold off
    
    figure(2)
    plot(detrend(ang_tan),detrend(dispInterp))
    drawnow
end



%%


%% Find the angle of the wing at each frame
x_init = 230;
y_init = 105;
wind = -3:3;

thresh = 100;
% yval = [];
[xmn, ymn] = meshgrid(wind);

clf
axis([0,10000,-inf,inf])
clear ang_tan

for kk=6600:2:mm.NumFrames
    
    % Get frame
    tmp = double(read(mm, kk));
    
    tmptmp = tmp(round(y_init) + wind, round(x_init) + wind);
    tmptmp = tmptmp.*(tmptmp > thresh);
    
    [mx, idx] = max(tmptmp(:));
    [yy,xx] = ind2sub(size(tmptmp), idx(1));
    
    xcent = sum(reshape(xmn.*tmptmp,1,[]))./sum(reshape(tmptmp,1,[]));
    ycent = sum(reshape(ymn.*tmptmp,1,[]))./sum(reshape(tmptmp,1,[]));
    
    x_init = round(x_init) + xcent;
    y_init = round(y_init) + ycent;
    yval(kk) = y_init;
    
     %Plotting
    clf
    imagesc(tmp);
    hold on;
    plot(x_init, y_init, 'ro');
%     scatter(X_cent,Y_cent,200,'.r')
%     scatter(centroid(1),centroid(2),200,'.g')
%     line([X_cent centroid(1)],[Y_cent centroid(2)]);
    axis(ax);
    drawnow
    
    kk
%     plot(ang_tan)
%     axis([0,10000,-inf,inf])
%     drawnow
end


%%

synchDisp = detrend(dispMM(80001:120000),0);
synchT   = dispTime(80001:120000)-8;


synchAng    = detrend(ang_tan(6001:end),0);
synchAT     = t(6001:end);

clf
subplot(1,2,1)
yyaxis left
plot(t,detrend(ang_tan,0)*180/pi);
ylabel("Angle (degrees)")
axis([-2,2,-25,25])
yyaxis right
plot(dispTime-7,detrend(dispMM,0))
ylabel("Disp (mm)")
axis([-2,2,-0.12,0.12])

downDispMM = interp1(dispTime-7,dispMM,t);

linefit = polyfit(detrend(ang_tan,0),detrend(downDispMM,0),1);

fitT    = linspace(-0.4,0.4,100);
slope = fitT(2)
fitLine = polyval(linefit,fitT);


subplot(1,2,2)
hold on
scatter(detrend(ang_tan(1:5000),0),detrend(downDispMM(1:5000),0),'.');
scatter(detrend(ang_tan(5001:10001),0),detrend(downDispMM(5001:10001),0),'.');
plot(fitT,fitLine,'k','LineWidth',2)
xlabel('Angle (degrees)')
ylabel('Disp(mm)')
legend('Synch','Asynch','Fit Line: 0.0965x+ 0')




