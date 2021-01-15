lost_spikes = 0; % total number of spikes ignored by the compression
h = waitbar(0,'Initializing waitbar...');
for class=0:9
    directory_src = strcat(num2str(class),'/');
    % handle the output file:
    directory_des_train = sprintf('CIFAR_DVS_100/train/%s', num2str(class));
    directory_des_test = sprintf('CIFAR_DVS_100/test/%s', num2str(class));
    Readfiles = dir(fullfile(directory_src,'*.aedat'));
    file_num = length(Readfiles);
    if ~exist(directory_des_train, 'dir')
        mkdir(directory_des_train);
    end
    if ~exist(directory_des_test, 'dir')
        mkdir(directory_des_test);
    end

    % for samples in the same class
    for ii=1:file_num
        TD =dat2mat(strcat(directory_src, Readfiles(ii).name));         
        % remove the possible points that are out of bound
        x = floor(TD(:, 4)/3);
        y = floor(TD(:, 5)/3);
        p = TD(:, 6);
        ts = floor(TD(:, 1)/12000);
        
        % the binary spike matrix:
        mat = zeros(42 * 42 * 2, 100);
        for ind = 1:length(ts)
            if p(ind) == 1
                ps = 1;
            else 
                ps = 0;
            end
            if x(ind) < 42 && y(ind)<42 && ts(ind) < 100
                mat(x(ind) + y(ind) * 42 + ps * 42 * 42 + 1, ts(ind)+1) = 1;
            end
        end
        % dump each channels to the file
        if ii < 900
            name = directory_des_train + "/" + num2str(ii)+".csv";
        else
            name = directory_des_test + "/" + num2str(ii)+".csv";
        end
        writematrix(mat, name);
        perc = (ii + class * 1000)/100;
        waitbar(perc/100, h, sprintf('Processing %.2f%% ...',perc));
    end
end
close(h);
    
