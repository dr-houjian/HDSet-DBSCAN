% This is the code of the HDSet-DBSCAN algorithm proposed in
% Jian Hou, Huaqiang Yuan, Marcello Pelillo. Game-Theoretic Hypergraph
% Matching with Density Enhancement. Pattern Recognition, vol. 133, 2023.

function demo_hdset_dbscan()

    %parameters
    nNN=100;
    thk=2;       %MinPts
    sigma=5;

    %read image and features
    fimg1='img1.ppm';
    fimg2='img2.ppm';
    img1=imread(fimg1);
    img2=imread(fimg2);
    
    para.th_p1=15;
    para.th_e1=3;
    para.th_p2=15;
    para.th_e2=3;
    para.th_d=10;
    [P1,P2,descr1,descr2]=img2feat(img1,img2,para);
    
    %build hsima
    nT=1;        %fixed
    [hsima cmatch time1]=hdset2hsima_img(P1,P2,descr1,descr2,nT,nNN,sigma);

    %do matching
    [label time2 time3]=hsima2match_dbscan(hsima,cmatch,thk);

    t_basic=time1+time2;
    t_extend=time1+time3;
    
    %show matching result
    match_c=cell(1,2);
    match_c(1)={cmatch(label==1,:)};
    match_c(2)={cmatch(label>1,:)};
    
    disp_match_multi(match_c,img1,img2,P1,P2,1);
    
    nmatch1=size(cmatch(label==1,:),1);
    nmatch2=size(cmatch(label>0,:),1);
    
    fprintf('\n nmatch1=%d nmatch=%d time1=%f time=%f\n',nmatch1,nmatch2,t_basic,t_extend);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This is used to generate the features used in matching.
function [P1,P2,descr1,descr2]=img2feat(img1,img2,para)

    th_p1=para.th_p1;
    th_e1=para.th_e1;
    th_p2=para.th_p2;
    th_e2=para.th_e2;
    th_d=para.th_d;

    [height1,width1]=size(img1);
    [height2,width2]=size(img2);

    [P1,descr1]=img2hmp(img1,th_p1,th_e1);
    [P2,descr2]=img2hmp(img2,th_p2,th_e2);
    [P1,descr1]=pts_filter(P1,descr1,th_d,height1,width1);
    [P2,descr2]=pts_filter(P2,descr2,th_d,height2,width2);

end

%This is used to generate the feature points and descriptors used in
%hypergraph matching.
%img: the image from imread
%pts: N*2, each row is the x,y of a point
%descr: N*m, each row is the descriptor of a point
function [pts,descr]=img2hmp(img,th_p,th_e)

    nchannel=size(img,3);
    if nchannel==3
        img=single(rgb2gray(img));
    else
        img=single(img);
    end
    
    %SIFT feature extraction with the vlfeat library
    [frames,descr]=vl_sift(img,'PeakThresh',th_p,'EdgeThresh',th_e);
    pts=frames(1:2,:);
    pts=pts';
    descr=descr';

    %post-processing, remove the features in the same position
    npts=size(pts,1);
    for i=1:npts-1
        for j=i+1:npts
            if i<=npts && j<=npts && sum(abs(pts(i,:)-pts(j,:)))==0
                pts(j,:)=[];
                descr(j,:)=[];
                j=j-1;
                npts=size(pts,1);
            end
        end
    end
end

function [P1,descr1]=pts_filter(P1,descr1,th,height,width)

    i=1;
    while 1>0
        pt1=P1(i,:);
        
        j=i+1;
        while 1>0
            pt2=P1(j,:);
            d=sqrt(sum((pt1-pt2).^2));
            
            if d<th
                P1(j,:)=[];
                descr1(j,:)=[];
                j=j-1;
            end
            
            j=j+1;
            
            if j>size(P1,1)
                break;
            end
        end
        
        i=i+1;
        
        if i>=size(P1,1)
            break;
        end
    end
    
    i=1;
    while 1>0,
        if P1(i,1)<20 || P1(i,2)<20 || P1(i,1)>width-20 || P1(i,2)>height-20
            P1(i,:)=[];
            descr1(i,:)=[];
        end
        i=i+1;
        
        if i>size(P1,1)
            break;
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This is used to generate the hsima used in hdset matching between images.
function [hsima cmatch interval]=hdset2hsima_img(P1,P2,descr1,descr2,nT,nNN0,scale)

    if size(P1,1)==2
        P1=P1';
    end
    if size(P2,1)==2
        P2=P2';
    end
    
    tic;
    
    %triangles
    [t1,t2,feat_t1,feat_t2]=pts2tri(P1,P2,nT);

    %sparsify
    nP2=size(P2,1);
    nT2=nP2*(nP2-1)*(nP2-2);
    if nNN0<=1
        nNN=floor(nT2*nNN0);
    else
        nNN=min(nT2,nNN0);
    end
    nNN=max(nNN,5);
    [ct1 ct2 hsim]=tri2ct_img(feat_t1,feat_t2,descr1,descr2,t1,t2,nNN,scale);

    %hsima
    nP1=size(P1,1);
    nP2=size(P2,1);
    
    [hsima,cmatch]=ct2hsima(ct1,ct2,hsim,nP1,nP2);
    
    interval=toc;
  
end

%This is used to generate the triangles and corresponding descriptions,
%used in hdset matching.
function [t1,t2,feat_t1,feat_t2]=pts2tri(P1,P2,nT)

    %generate triangles
    t1=generate_t1(P1,nT);
    t2=generate_t2(P2);
    
    %generate descriptions of triangles
    feat_t1=feature_sin(t1,P1);
    feat_t2=feature_sin(t2,P2);
end

%This is used to generate the triangles in a set of data, used in
%hypergraph matching.
function t1=generate_t1(P1,nT)
    if ~exist('nT','var')
        nT=1;
    end
    
    nP1=size(P1,1);
    
    if nT<=1
        npts=round((nP1-1)*nT);
    else
        npts=min(nP1-1,nT);
    end

    dima0=pdist(P1,'euclidean');
    dima=squareform(dima0);
    [~,idx_sort]=sort(dima,2,'ascend');
    t1=zeros(round(nP1*npts*(npts-1)/2),3);
    
    count=0;
    for i=1:nP1
        ss=idx_sort(i,2:npts+1);
        t10=combntns(ss,2);
        nt10=size(t10,1);
        t11=ones(nt10,1)*i;
        t12=[t11,t10];
        t1(count+1:count+nt10,:)=t12;
        count=count+nt10;
    end
    
    t1=sort(t1,2,'ascend');
    t1=unique(t1,'rows');
    
end

%This is used to generate the triangles in reference image, used in
%hypergraph matching.
function t2=generate_t2(P2)

    nP2=size(P2,1);
    
    ss=1:nP2;
    t2=combntns(ss,3);

end

function feat=feature_sin(tri,P)

    nt=size(tri,1);
    feat=zeros(nt,3);
    for i=1:nt
        vec=tri(i,:);
        v1=P(vec(1),:);
        v2=P(vec(2),:);
        v3=P(vec(3),:);
        d12=sqrt(sum((v1-v2).^2));
        d23=sqrt(sum((v2-v3).^2));
        d13=sqrt(sum((v1-v3).^2));
                
        cos1=(d12^2+d13^2-d23^2)/(2*d12*d13);
        cos2=(d12^2+d23^2-d13^2)/(2*d12*d23);
        cos3=(d13^2+d23^2-d12^2)/(2*d13*d23);
                
        feat(i,:)=[(1-cos1^2)^0.5 (1-cos2^2)^0.5 (1-cos3^2)^0.5];
    end

end

%This is build the similarity between triangles.
function [ct1 ct2 hsim]=tri2ct_img(feat_tri1,feat_tri2,feat_vertex1,feat_vertex2,t1,t2,nNN,sigma)

    %new, fast
    if size(feat_tri1,1)<size(feat_tri1,2)
        feat_tri1=feat_tri1';
    end
    if size(feat_tri2,1)<size(feat_tri2,2)
        feat_tri2=feat_tri2';
    end
    
    t22=t2(:,[1 3 2]);
    t23=t2(:,[2 1 3]);
    t24=t2(:,[2 3 1]);
    t25=t2(:,[3 1 2]);
    t26=t2(:,[3 2 1]);
    t2=[t2;t22;t23;t24;t25;t26];
    
    feat_tri22=feat_tri2(:,[1 3 2]);
    feat_tri23=feat_tri2(:,[2 1 3]);
    feat_tri24=feat_tri2(:,[2 3 1]);
    feat_tri25=feat_tri2(:,[3 1 2]);
    feat_tri26=feat_tri2(:,[3 2 1]);
    feat_tri2=[feat_tri2;feat_tri22;feat_tri23;feat_tri24;feat_tri25;feat_tri26];

    [inds, dists] = annquery(feat_tri2', feat_tri1', nNN, 'eps', 10); %with the ann_mwrapper library
    inds=inds';
    dists=dists';
    sigma=mean(dists(:))*sigma;
   
    %build hsim
    nt1=size(t1,1);
    ct1=zeros(nt1*nNN,3);
    ct2=zeros(nt1*nNN,3);
    hdist=zeros(nt1*nNN,1,'single');
    
    sima_sift=slmetric_pw(single(feat_vertex1'),single(feat_vertex2'),'nrmcorr'); %with the pwmetric library
    
    [ii,jj]=find(sima_sift>0.85);
    iijj=[ii,jj];

    for i=1:nt1
        i0=(i-1)*nNN+1;
        i1=i*nNN;
        ct1(i0:i1,:)=repmat(t1(i,:),nNN,1);
        ct2(i0:i1,:)=t2(inds(i,:),:);
        hdist(i0:i1)=dists(i,:);
    end
    
    cm1=[ct1(:,1),ct2(:,1)];
    cm2=[ct1(:,2),ct2(:,2)];
    cm3=[ct1(:,3),ct2(:,3)];

    idx1=ismember(cm1,iijj,'rows');
    idx2=ismember(cm2,iijj,'rows');
    idx3=ismember(cm3,iijj,'rows');

    idx=logical(idx1.*idx2.*idx3);
    ct1=ct1(idx,:);
    ct2=ct2(idx,:);
    hdist=hdist(idx);
        
    hsim=exp(-hdist/sigma);
    
    clear sima_sift;

end

%This is to build hsima based on the candiate triangles and similarities.
function [hsima,cmatch]=ct2hsima(ct1,ct2,hsim,nP1,nP2)

    [nct,order]=size(ct1);    %number of candidate triangle matches
    
    cmatch=zeros(nP1*nP2,2);
    k=1;
    for i=1:nP1
        for j=1:nP2
            cmatch(k,:)=[i,j];
            k=k+1;
        end
    end
    
    %obtain all the candidate matches and build the hyper-sima
    hsima=zeros(nct,order+1,'single'); %the idx of the three candidiate matches, and corresponding sim
    
    idx=zeros(1,order);       %the idx of the three candidate matches in a triangle in cmatch
    for i=1:nct               %all the candidate triangle matches
        for j=1:order         %all the candiate matches in a triangle
            idx(j)=(ct1(i,j)-1)*nP2+ct2(i,j);
        end
        hsima(i,1:order)=idx;
        hsima(i,order+1)=hsim(i);
    end

    clear hsim ct;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% do matching
function [label t_basic t_extend]=hsima2match_dbscan(hsima,cmatch,thk)

    th_weight=0.0001;
    toll=1e-3;
    max_iter=500;
    
    [nedge,dimen]=size(hsima);
    ndata=size(cmatch,1);
    
    H.nVertices=ndata;
    H.nEdges=nedge;
    H.edges=hsima(:,1:dimen-1);
    H.w=hsima(:,dimen);
    
    label=zeros(1,ndata);
    
    x=ones(ndata,1);
    x=x/sum(x);
    
    tic;

    nhdset=1;

    %do clustering
    [x,Fx,niter]=hgtExtractCluster(H,x,toll,max_iter);
    x=reshape(x,1,length(x));
        
    %enforce one-to-one constraint
    x=oto2x(x,cmatch);
        
    %the cluster
    idx_hdset=find(x>th_weight);

    label(idx_hdset)=1;
        
    t_basic=toc;
        
    [idx_hdset label]=extend_dbscan(hsima,label,nhdset,cmatch,thk);
        
    t_extend=toc;

end

% This is the code of
%
% S. R. Bulo, M. Pelillo, A game-theoretic approach to hypergraph clustering,
% IEEE Transactions on Pattern Analysis and 755 Machine Intelligence 35 (6)
% (2013) 1312¨C1327.
%
% Written by Samuel Rota Bulo.
%
% Hypergraph clustering
% 
% H: hypergraph structure
%    H.nVertices: number of vertices (N)
%    H.nEdges:    number of edges (M)
%    H.edges:     MxK matrix of indices to vertices representing the edges
%    H.w:         Mx1 vector edge weights
%
% x: Nx1 starting point in the simplex
%
% maxiter: maximum number of iterations

function [x,Fx,niter]=hgtExtractCluster(H,x,toll,maxiter)
  if ~exist('maxiter','var')
    maxiter=1000;
  end
  
  if size(x,2)>1
      x=x';
  end
  
  niter=0;
  error=2*toll;
  old_x=x;
  while niter<maxiter && error>toll
    Fx=zeros(H.nVertices,1);
    for i=1:H.nEdges
      edge=H.edges(i,:);
      tmp=prod(x(edge))*H.w(i);
      if tmp>0
        Fx(edge)=Fx(edge)+tmp./x(edge);
      end 
    end
    x=x.*Fx;
    xFx=sum(x);
    if xFx==0
      return;
    end
    x=x/xFx;
    
    error=norm(x-old_x);
    old_x=x;
    
    niter=niter+1;
    %fprintf('iter=%d error=%f\n',niter,error);
  end
end

function x=oto2x(x,cmatch)

    if size(x,1)>1
        x=x';
    end

    [~,idx_sort]=sort(x,'descend');
    vec_t=cmatch(idx_sort,1);
    
    vt=unique(cmatch(:,1));
    for i=1:length(vt)
        no=vt(i);
        idx=find(vec_t==no);
        if length(idx)>1
            x(idx_sort(idx(2:length(idx))))=0;
        end
    end
    
    vec_r=cmatch(idx_sort,2);
    vr=unique(cmatch(:,2));
    for i=1:length(vr)
        no=vr(i);
        idx=find(vec_r==no);
        if length(idx)>1
            x(idx_sort(idx(2:length(idx))))=0;
        end
    end

end

function [idx_hdset label]=extend_dbscan(hsima,label,nhdset,cmatch,thk)

    dimen=size(hsima,2);
    edges=hsima(:,1:dimen-1);
    sims=hsima(:,dimen);
    
    idx_hdset=find(label==nhdset);
    nsize=length(idx_hdset);

    %find the pair threshold, the sim with k-th nn
    v_thsim=find_thsim(edges,sims,idx_hdset,thk);
    
    if ~isempty(v_thsim)
        thsim=min(v_thsim);
        
        %list all the pairs in the hdset
        pair=[];
        for i=1:nsize-1
            for j=i+1:nsize
                vec=idx_hdset([i,j]);
                pair=[pair;vec];
            end
        end
    
        %start expansion
        while ~isempty(pair)
            vec=pair(1,:);
            pair(1,:)=[];
            ii=vec(1);
            jj=vec(2);
            [row1,col1]=find(edges==ii);
            [row2,col2]=find(edges==jj);
            row=intersect(row1,row2);
            sim=sims(row);
            idx_sim=find(sim>=thsim);
            idx_sim=reshape(idx_sim,1,length(idx_sim));
            row=reshape(row,1,length(row));
            if length(idx_sim)>thk
                for i=row(idx_sim)
                    tri=edges(i,:);
                    vv=[ii jj];
                    pt=setdiff(tri,vv);
                    
                    if length(pt)>1
                        printf('Wrong.');
                    end
                    
                    idx=pt;
                    
                    % one-to-one filtering
                    if label(idx)==0
                        idx1=find(cmatch(idx_hdset,1)==cmatch(idx,1), 1);
                        idx2=find(cmatch(idx_hdset,2)==cmatch(idx,2), 1);
                        
                        if isempty(idx1) && isempty(idx2)
                            label(idx)=nhdset+0.1;
                            idx_hdset=[idx_hdset,idx];
                        end
                    end
                end
            end
        end
    
    end

end

function v_thsim=find_thsim(edges,sims,idx_hdset,thk)

    nsize=length(idx_hdset);

    v_thsim=[];
    for i=1:nsize-1
        ii=idx_hdset(i);
        [row1,col1]=find(edges==ii);
        
        for j=i+1:nsize
            jj=idx_hdset(j);
            [row2,col2]=find(edges==jj);
            
            row=intersect(row1,row2);   %the edges including [i,j]
            sim=sims(row);
            sim0=sort(sim,'descend');
            
            if length(sim0)>=thk
                v_thsim=[v_thsim,sim0(thk)];
            end
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function disp_match_multi(match_c,img1,img2,P1,P2,flag_combine)

    if ~exist('flag_combine','var')
        flag_combine=1;
    end
    
    nlabel=length(match_c);
    match=[];
    for i=1:nlabel
        match=[match;match_c{i}];
    end
    label=zeros(1,size(match,1));

    npart=length(match_c);
    count=0;
    for i=1:npart
        match0=match_c{i};
        nmatch0=size(match0,1);
        label(count+1:count+nmatch0)=i;
        count=count+nmatch0;
    end
    
    nLineWidth=3;
    
    color=cell(1,7);
    color(1)={'b'};
    color(2)={'r'};
    color(3)={'g'};
    color(4)={'k'};
    color(5)={'m'};
    color(6)={'y'};
    color(7)={'c'};
    
    line=cell(1,3);
    line(1)={'-'};
    line(2)={'--'};
    line(3)={'-.'};
    
    %the combined image
    img3=img_combine(img1,img2,flag_combine);
    [h1,w1,~]=size(img1);
    
    imshow(img3,'Border','tight');
    hold on;
    
    %the feature points
    alpha=0:pi/20:2*pi;
    R=6;
    x0=R*cos(alpha);
    y0=R*sin(alpha);
    
    for i=1:size(P1,1)
        x=P1(i,1);
        y=P1(i,2);
        
        x1=x+x0;
        y1=y+y0;
        plot(x1,y1,'r-','LineWidth',1);
    end
    
    for i=1:size(P2,1)
        if flag_combine==1
            x=P2(i,1)+w1;
            y=P2(i,2);
        else
            x=P2(i,1);
            y=P2(i,2)+h1;
        end
        
        x1=x+x0;
        y1=y+y0;
        plot(x1,y1,'r-','LineWidth',1);
    end

    %matches
    alpha=0:pi/20:2*pi;
    R=6;
    x0=R*cos(alpha);
    y0=R*sin(alpha);
    
    for i=1:size(match,1)
        k=label(i);
        ii=floor((k-1)/7);
        jj=round(k-ii*7);
        ii=round(ii+1);
        symbol=[color{jj},line{ii}];
        
        idx1=match(i,1);
        idx2=match(i,2);
        
        if idx1==0 || idx2==0
            continue;
        end
        
        %feature
        x1=P1(idx1,1);
        y1=P1(idx1,2);
        plot(x1+x0,y1+y0,'r-','LineWidth',1);
        fill(x1+x0,y1+y0,'r');
        
        %match
        if flag_combine==1
            x2=P2(idx2,1)+w1;
            y2=P2(idx2,2);
        else
            x2=P2(idx2,1);
            y2=P2(idx2,2)+h1;
        end
        
        plot(x2+x0,y2+y0,'r-','LineWidth',1);
        fill(x2+x0,y2+y0,'r');
    
        plot([x1,x2],[y1,y2],symbol,'LineWidth',nLineWidth);
        hold on;
    end
    
    axis off;
    hold off;
    
    aframe=getframe(gcf);
    imwrite(aframe.cdata,'d:\match.jpg');

end

%This is used to merge two images into one, used to show matching results.
function img=img_combine(img1,img2,mode)

    if ~exist('mode','var')
        mode=1;
    end

    [h1,w1,nc1]=size(img1);
    [h2,w2,nc2]=size(img2);

    if mode==1
        if h1>h2
            img20=zeros(h1,w2,nc2,'uint8');
            img20(1:h2,1:w2,1:nc2)=img2;
            img=[img1,img20];
        elseif h1<h2
            img10=zeros(h2,w1,nc1,'uint8');
            img10(1:h1,1:w1,1:nc1)=img1;
            img=[img10,img2];
        else
            img=[img1,img2];
        end
    else
        if w1>w2
            img20=zeros(h2,w1,nc2,'uint8');
            img20(1:h2,1:w2,1:nc2)=img2;
            img=[img1;img20];
        elseif w1<w2
            img10=zeros(h1,w2,nc1,'uint8');
            img10(1:h1,1:w1,1:nc1)=img1;
            img=[img10;img2];
        else
            img=[img1;img2];
        end
    end

end