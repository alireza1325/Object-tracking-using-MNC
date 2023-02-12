%% set up video
clc,close all, clear all;
format short 

videoReader = vision.VideoFileReader('Noon_traffic.mp4');  % Read sample matlab video
writerObj = VideoWriter('OpticalFlow4.mp4','MPEG-4');        % Create video writer object
writerObj.FrameRate = 30;                                   % Set the fps (Optional)
open(writerObj);                                            % Open the video writer
frameStartTraining = 1;
frameStart = 51;
frameEnd = 250;
fcount = 0;

% machine learning detects changing part of each frame
foregroundDetector = vision.ForegroundDetector('NumGaussians', 5,'NumTrainingFrames', 50); 
n_tri_MNC = zeros(frameEnd,1);

MNC_divs = zeros(frameEnd-frameStart,100);
MNC_curls = zeros(frameEnd-frameStart,100);

grad = 0;
div = 0;

x_cw = 0;
for i = 1:20
    x_cw = [x_cw,[(-(-1)^i)*i:(-1)^i:(-1)^i*i]];
end
x_cw = x_cw(2:end);
y_cw = [-1,-2,-1,0,1,2,1,0,-1,-2,-3,-4,-3,-2,-1,0,1,2,3,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0];
figure
plot(x_cw(1:length(y_cw)),y_cw,'-*')
xlabel('x')
ylabel('y')
title('Clockwise optical flow for negative curl')

x_cw = 0;
for i = 1:20
    x_cw = [x_cw,[(-(-1)^i)*i:(-1)^i:(-1)^i*i]];
end
x_cw = x_cw(2:end);
y_cw = [-1,-2,-1,0,1,2,1,0,-1,-2,-3,-4,-3,-2,-1,0,1,2,3,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0];
figure
plot(y_cw,x_cw(1:length(y_cw)),'-*')
xlabel('x')
ylabel('y')
title('CounterClockwise optical flow for positive curl')


for count = frameStartTraining:frameEnd                  % The number of frames to process
    disp(count)                         % Display the current frame number
    frame = step(videoReader);          % Read the next video frame
    frame = imresize(frame,0.5);
    
    %frame = frame*1.5;
    frameGray = rgb2gray(frame);        % Convert the frame to grayscale
    frameSize = size(frameGray);        % Get the dimensions of the frame
    %if count ==1
    %    bkFrame = frameGray;            % Get the first frame as background frame
    %end
    foreground = step(foregroundDetector, frameGray);
    if count>=frameStart
        fcount = fcount+1;
                     
        frameBW2 = bwareaopen(foreground,100);   % Remove blobs smaller than 50 (Turn dark foreground to white)
        se = strel('disk',10);                  % Use a disk of radius 10 to dialte and erode object shape
        frameBW3 = imdilate(frameBW2,se);       % Dialate the object shape
        %frameBW3 = imerode(frameBW3,se);        % Erode the object shape
        
        s = regionprops(frameBW3,'centroid');   % Get the stats of the blobs
        blbCentroids = cat(1,s.Centroid);       % Exatract the centroids
        
        %[B] = bwboundaries(frameBW3);           % Save the object boundaries
        corners = corner(frameBW3);             % Detect corners on edges
                       
        figure('visible','on','Position', [10 10 1500 1500]); 
        subplot(1,2,1)
        imshow(frame), axis on, title(['Triangulated frame with MNC optical flow ',num2str(count)],'Color','blue'); % Plot on top of the current frame 

        hold on
        seedPoints = [blbCentroids;corners(:,:)];      
                                         
        MNC_div = 0;
        MNC_curl = 0;
        C=0;
        D=0;
        %% Delaunay triangulation
        if size(seedPoints,1)>3
            delaunay_tri=delaunay(seedPoints(:,1),seedPoints(:,2)); % Generating delaunay triangles        
            % Extract the triangle vertices
            x_tri=[]; % Clear the variable ( Not to redraw triangles from earlier frame)
            y_tri=[];
            facecntr =[];
            for i=1:size(delaunay_tri,1)
                x_tri(:,i)=seedPoints(delaunay_tri(i,:),1)';
                y_tri(:,i)=seedPoints(delaunay_tri(i,:),2)';
            end           
            % Draw the triangles
            faceclr='green';           % False color the triangles
            faceopac=0.2;               % Triangle color opacity
            vrtxmrkr='o';               % Triangle vertex marker type
            vrtxmrkrsize=3;             % Triangle vertex marker size
            vrtxmrkrclr='yellow';            % Triangle vertex marker fill color (Boundary points)
            mrkredgeclr='white';        % Tringle vertex edge color
            edgeclr=[0.3 0.3 0.3];      % Triangle edge color
            edgewidth=1;                % Triangle edge width         
            patch(x_tri,y_tri,faceclr,'FaceAlpha',faceopac,'Marker',vrtxmrkr,'MarkerFaceColor',vrtxmrkrclr ...
                ,'MarkerSize',vrtxmrkrsize,'MarkerEdgeColor',mrkredgeclr,'EdgeColor',edgeclr,'LineWidth',edgewidth);                
   
            % Finding the MNC
            MNCLoc = 0; % Clear variables
            MNCVal = 0;
            distMat =[];
            for i = 1: size(blbCentroids,1) % Go through blobs one by one
                [row, column] = find(delaunay_tri == i); % Find all the triangles associated with current blob centroid
                if length(row)>MNCVal       % If the number is more then this is the new MNC
                    MNCLoc = i;             % Save the MNC location
                    MNCVal = length(row);   % Save the number of new MNC triangles
                elseif row==MNCVal          % There can be two or more MNCs
                    MNCLoc = [MNCLoc ; i];  % Save them as well
                end
                
                % Draw the MNC triangualation and the vertices
                faceclr='yellow';              % False color the triangles
                faceopac=0.3;               % Triangle color opacity
                vrtxmrkr='o';               % Triangle vertex marker type
                vrtxmrkrsize=5;             % Triangle vertex marker size
                vrtxmrkrclr='r';            % Triangle vertex marker fill color (Boundary points)
                mrkredgeclr=[0 0 0];        % Tringle vertex edge color
                edgeclr=[0.3 0.3 0.3];      % Triangle edge color
                edgewidth=1;                % Triangle edge width

                div = 0;
                curl =0;
                MNC_V_x = zeros(3,1);
                MNC_V_y = zeros(3,1);
                
                for ii = 1: size(MNCLoc,1)
                    NowMNC = MNCLoc(ii);
                    [MNCR,MNCC] = find(delaunay_tri == NowMNC);
                    for i3 = 1:size(MNCR,1)
                        x = seedPoints(delaunay_tri(MNCR(i3),:),1);
                        y = seedPoints(delaunay_tri(MNCR(i3),:),2);
                        

                        for j = 1:length(y)
                            row = floor(y(j));
                            col = floor(x(j));
                            pdy = double(frame(row+1,col)) - double(frame(row,col));
                            pdx = double(frame(row,col+1)) - double(frame(row,col));
                            div = [div,pdy + pdx];      % calculate the divergent of MNC vertices
                            curl = [curl,pdy - pdx];      % calculate the curl of MNC vertices
                            C = [C,curl];
                            D = [D,div];
                            
                            %text(col,row,['\leftarrow ',num2str(curl)],'Color','blue','FontSize',11);
                        end
                        MNC_V_x = [MNC_V_x,x];
                        MNC_V_y = [MNC_V_y,y];
                    end

                    div = mean(unique(div));
                    curl = mean(unique(curl));
                end
                
                MNC_V_x = floor(MNC_V_x(:,2:end));
                MNC_V_y = floor(MNC_V_y(:,2:end));
                min_x = min(min(MNC_V_x));
                max_x = max(max(MNC_V_x));
                min_y = min(min(MNC_V_y));
                max_y = max(max(MNC_V_y));
                patch(MNC_V_x,MNC_V_y,faceclr,'FaceAlpha',faceopac,'Marker',vrtxmrkr,'MarkerFaceColor',vrtxmrkrclr,'MarkerSize',vrtxmrkrsize,'MarkerEdgeColor',mrkredgeclr,'EdgeColor',edgeclr,'LineWidth',edgewidth);
                
                xx = MNC_V_x(1,1);
                yy = MNC_V_y(1,1);
                curl = double(frame(yy+1,xx)) - double(frame(yy,xx+1));
                hold on
                neg=1;
                pos=1;
                
                for i=1:150
                    if(curl<0)
                        pos = 1;
                        xx = xx+x_cw(neg);
                        yy = yy+y_cw(neg);
                        plot(xx,yy,'-o','MarkerFaceColor','r', 'MarkerEdgeColor','r')
                        %text(xx,yy,['\leftarrow ',num2str(curl)],'Color','blue','FontSize',12);
                        neg = neg+1;
                    else
                        neg = 2;
                        xx = xx+y_cw(pos);
                        yy = yy+x_cw(pos);
                        plot(xx,yy,'-o','MarkerFaceColor','g', 'MarkerEdgeColor','g')
                        %text(xx,yy,['\leftarrow ',num2str(curl)],'Color','blue','FontSize',10);
                        pos = pos+1;
                    end
                    if (xx>max_x || xx<min_x || yy>max_y || yy<min_y)
                        break
                    end
                    curl = double(frame(yy+1,xx)) - double(frame(yy,xx+1));

                end
                

                MNC_div = [MNC_div,div];
                MNC_curl = [MNC_curl,curl];
            end
            
            subplot(1,2,2)
            patch(MNC_V_x,MNC_V_y,faceclr,'FaceAlpha',faceopac,'Marker',vrtxmrkr,'MarkerFaceColor',vrtxmrkrclr,'MarkerSize',vrtxmrkrsize,'MarkerEdgeColor',mrkredgeclr,'EdgeColor',edgeclr,'LineWidth',edgewidth);
            title(['Optical flow of MNC ',num2str(count)],'Color','blue');
            hold on
            
            xx = MNC_V_x(1,1);
            yy = MNC_V_y(1,1);
            curl = double(frame(yy+1,xx)) - double(frame(yy,xx+1));
            neg=1;
            pos=1;
                
            for i=1:150
                 if(curl<0)
                        pos = 1;
                        xx = xx+x_cw(neg);
                        yy = yy+y_cw(neg);
                        plot(xx,yy,'-o','MarkerFaceColor','r', 'MarkerEdgeColor','r')
                        text(xx,yy,['\leftarrow ',num2str(curl)],'Color','blue','FontSize',10);
                        neg = neg+1;
                    else
                        neg = 2;
                        xx = xx+y_cw(pos);
                        yy = yy+x_cw(pos);
                        plot(xx,yy,'-o','MarkerFaceColor','g', 'MarkerEdgeColor','g')
                        text(xx,yy,['\rightarrow ',num2str(curl)],'Color','blue','FontSize',10);
                        pos = pos+1;
                    end
                    curl = double(frame(yy+1,xx)) - double(frame(yy,xx+1));
                    if (xx>max_x || xx<min_x || yy>max_y || yy<min_y)
                        break
                    end
            end
            
            for i = 1:length(MNC_V_x(:,1))
                for j = 1:length(MNC_V_x(1,:))
                    row = MNC_V_y(i,j);
                    col = MNC_V_x(i,j);
                    pdy = double(frame(row+1,col)) - double(frame(row,col));
                    pdx = double(frame(row,col+1)) - double(frame(row,col));
                    %div = pdy + pdx;      % calculate the divergent of MNC vertices
                    curl = pdy - pdx;      % calculate the curl of MNC vertices
                    text(col,row,['\leftarrow ',num2str(curl)],'Color','magenta','FontSize',10);
                end
            end
               
            
        
        MNC_div = unique(MNC_div);
        MNC_curl = unique(MNC_curl);

        end
        hold off
        
        writeFrame = getframe(gcf);             % Capture the current displayed frame
        close();                                % Close the figure
        writeVideo(writerObj,writeFrame.cdata); % Write the captured frame to the video
        
        
        MNC_divs(count-frameStart+1,1:length(MNC_div)) = MNC_div;
        MNC_curls(count-frameStart+1,1:length(MNC_curl)) = MNC_curl;
    end
end
close(writerObj);   % Close the video writer object


close all;

colors=colormap(jet(50));
r = randi([1 50],1,frameEnd-frameStart+1);


%======================================================================
%                                                                     %
%               Generating plot for Mean of divergent                 %
%                                                                     %
%======================================================================
figure('visible','on','Position', [10 10 1200 800])
for i = 1:length(MNC_divs(:,1))
    Y = MNC_divs(i,:);
    Y = Y(Y~=0);
    X = (i+frameStart-1)*ones(length(Y),1);
    plot(X,Y,'o','MarkerFaceColor',colors(r(i),:), 'MarkerEdgeColor',colors(r(i),:))
    hold on
end
xlabel('Frame No.')
ylabel('Mean of MNCs vertices divergent (lx)')
title('MNC divergence')
%ylim([0,3])
grid on
hold off

%======================================================================
%                                                                     %
%               Generating plot for Mean of Curls                     %
%                                                                     %
%======================================================================
figure('visible','on','Position', [10 10 1200 800])
for i = 1:length(MNC_curls(:,1))
    Y = MNC_curls(i,:);
    Y = Y(Y~=0);
    X = (i+frameStart-1)*ones(length(Y),1);
    plot(X,Y,'o','MarkerFaceColor',colors(r(i),:), 'MarkerEdgeColor',colors(r(i),:))
    hold on
end
xlabel('Frame No.')
ylabel('Mean of MNCs vertices curl (lx)')
title('MNC Curl')
%ylim([0,3])
grid on
hold off