function file = get_dirfiles(path,name_filter)
% filter files in current dir and subdir of given path

file=filt_file(fullfile(path,name_filter));
subpath = dir(path);
for i=1:length(subpath)-2
    sspath = fullfile(path,subpath(i+2).name);
    if isdir(sspath)
        file = [file get_dirfiles(sspath,name_filter)];
    end    
end

end


function refile = filt_file(name)
% filter files for given fullfile with filtering
[fpath,fname,fext]=fileparts(name);
temp=dir(name);
refile=[];
for i=1:length(temp)
    ttemp = fullfile(fpath,temp(i).name);
    if ~isdir(ttemp)
    refile{i}=ttemp;
    end
end
end