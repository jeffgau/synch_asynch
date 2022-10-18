clear
system = 'moth'; % moth, flapper, or robobee
tic % Typically ~0.5-1s per moth simulation

%% Load System and Simulation parameters as Map containers
sys_param = load_system_param(system);

sim_param = load_simulation_param(system, sys_param('f_n'));


%% Load datastruct from external hard drive
% File is too large to store in git. Ask James to share it if necessary

% Load datastruct
load('D:\MATLABdata\100422_datastruct.mat');

%%
ntests      = sim_param('ntests');
r3_range    = sim_param('r3_range');
sampfreq    = sim_param('sampling_f');

[t_0,yax]   = convertr3t0(r3_range,sim_param('r4_ratio'),sys_param('f_n'));

% preallocate spectral matrix. 
spectra     = zeros(ntests,ntests,10001);

%% compute spectra
% r: loops over r3
% k: loops over Kr

for r = 1:ntests
    for k = 1:ntests
        r,k
        % Pull wing angle and assign a time vector
        position    = datastruct.lc_array(r,k).position;
        time        = 0:1/sampfreq:(1-1/sampfreq);

        % Plotting
        clf
        plot(time,position)
%         pause()
        
        % Get the FFT of flapping angle
        [freq, peaks] = fourier_analysis(position,sampfreq);

        clf
        stem(freq,peaks,'.')
        set(gca,'XScale','log')
        
        % Save FFT magnitudes. Freq will be the same for all, so not 
        % necessary to save each time
        spectra(r,k,:) = peaks;

        % Power density, i.e. how much of the total spectrum is captured by
        % the single maximum frequency
        [maxmag, maxind] = max(peaks);
        maxfreq(r,k)    = freq(maxind);
        sumpeaks        = sum(peaks);
        
        % percent of total spectrum captured by the maximum frequency.
        % Basically an expression of the spread of the spectrum
        powden(r,k) = maxmag/sumpeaks;  

    end
end


%% 
% close all
numfreqs = 200;

figure(1)
clf
% figure(2)
colormap(powerpurple)
colorbar

Kr = linspace(0,1,100);

savevid = 1;

if savevid
    v = VideoWriter("additional_analysis\spectral_analysis_oct18",'MPEG-4');
    v.FrameRate = 10;
    open(v)
end

for i = 1:100

    specmat = squeeze(spectra(i,:,1:numfreqs));
    
    % Kr at the synch-asynch boundary
    transitidx = find(abs(datastruct.freq_array(i,:)-1)<0.001,2);
    Krtrans = Kr(transitidx(2));

    f1 = figure(1);
    f1.Color = 'white';
    clf
    subplot(1,2,1)
    surf(freq(1:numfreqs),Kr,specmat,'EdgeAlpha',0)
    grid off
    set(gca,'Layer','top')
    hold on
    yline(Krtrans,'Color',[.4 .4 .4])
    view(90,-90)
%     colormap('bone')
    axis square
    ylabel("K_r")
    xlabel("Frequency (Hz)")
    title(sprintf("Frequency Spectrum, t_0/T_n = %1.3f",yax(i)))
%     drawnow    

    subplot(1,2,2)
    surf(Kr,yax(1:end),datastruct.conv_array(:,:),'edgecolor', 'none','HandleVisibility','off')
    % pcolor(synch_gain_range,yax(1:end),datastruct.conv_array(:,:))
    axis square
    axis([0 1 min(yax) max(yax)])
    view(0,90)
    xlabel('K_r')
    ylabel('t_o/T_n')    
    a = colorbar;
    a.Label.String = 'power (au)';
    a.Label.FontSize = 14;
    %set(gca,'ColorScale','log')
    set(gca,'YScale','log')
    set(gca, "Layer","top")
    hold on
    fcon = contour3(Kr,yax(1:end),datastruct.freq_array(:,:)+80,80+[0.99 1],'Color',[.4 .4 .4]);
    plot3([0 1], [yax(i) yax(i)], [100, 100],'r-','LineWidth',2)
    legend(["Asynch Boundary","current slice"])
    
%     f2 = figure(2);
%     hold on
%     plot(linspace(0,1,100),max(specmat'))


    drawnow
    if savevid
        writeVideo(v,getframe(f1));
    end
%     pause(0.01)
end

if savevid
    close(v)
end
%%

% figure()
clf
colormap("hot")
surf(linspace(0,1,100),yax,powden,"EdgeAlpha",0)
% pcolor(synch_gain_range,yax(1:end),datastruct.conv_array(:,:))
axis square
axis([0 1 min(yax) max(yax)])
view(0,90)
xlabel('K_r')
ylabel('t_o/T_n')
a = colorbar;
a.Label.String = 'p';
a.Label.FontSize = 14;
%set(gca,'ColorScale','log')
set(gca,'YScale','log')
set(gca, "Layer","top")

%%




