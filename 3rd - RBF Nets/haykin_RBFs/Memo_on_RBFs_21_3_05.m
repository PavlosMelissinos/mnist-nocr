%______________ simulating the RBF_net solution to the XOR-problem

X_data=[1 1 ; 0 0; 1 0 ; 0 1]; d=[0 0 1 1]'; % the dataformat is row-vectors  
plot(X_data(:,1),X_data(:,2),'*',X_data(d==1,1),X_data(d==1,2),'ro'),grid,axis([-0.5 1.5 -0.5 1.5])


centers=[1 1; 0 0] % selecting the centers


exp(-(norm(X_data(1,:)-centers(1,:)))^2) % exp(-||X1-c1||^2)
exp(-(norm(X_data(2,:)-centers(1,:)))^2) % exp(-||X2-c1||^2)
% .....
exp(-(norm(X_data(3,:)-centers(2,:)))^2) % exp(-||X3-c2||^2)
exp(-(norm(X_data(4,:)-centers(2,:)))^2) % exp(-||X4-c2||^2)
%the above (4 x 2) activations are tabulated in G as follows


G=[]; for j=1:4,for i=1:2,G(j,i)=exp(-(norm(X_data(j,:)-centers(i,:)))^2);,end, end
G
G=[G,[1;1;1;1]] % including biases

%______train the network
 w=inv(G'*G)*G'*d % the estimated weights
%_________________________________________


 


 X_in=X_data(1,:);

%_____running the network
f1=exp(-(norm(X_in-centers(1,:)))^2); f2=exp(-(norm(X_in-centers(2,:)))^2); % the hidden layer
y_out=[f1 f2 1]*w;  
 
X_in=X_data(2,:); f1=exp(-(norm(X_in-centers(1,:)))^2); f2=exp(-(norm(X_in-centers(2,:)))^2);
                  X_in,  y_out=[f1 f2 1]*w  

X_in=X_data(3,:); f1=exp(-(norm(X_in-centers(1,:)))^2); f2=exp(-(norm(X_in-centers(2,:)))^2);
                  X_in,  y_out=[f1 f2 1]*w  
                                    
  
X_in=X_data(4,:); f1=exp(-(norm(X_in-centers(1,:)))^2); f2=exp(-(norm(X_in-centers(2,:)))^2);
                  X_in,  y_out=[f1 f2 1]*w  
                
% exercise: repeat  the above steps without biases or repeat with different centers ..... 

                  
                  
                  
                  
%____________Haykin's demonstration - classification example

% 1. make the data
   P=mk_data(200); list=P(:,5)==1; 
  figure(1),clf
  subplot(1,2,1), plot(P(:,1),P(:,2),'.',P(list,1),P(list,2),'r.'),grid % plot the two-class 2D data

% 2. Train the RBF-network
  w = rbf(P(1:100,1:2), P(:,1:2), P(1:100,3:4), 4, 1); %%%% IMPORTANT NOTE---> the role of regularization parameter lamda

% 3. make a test set
   T=mk_data(500);

% 4. compute the network outputs with the test set as input
   rbfout=rbf_test(w,T(:,1:2),P(:,1:2),4);

% 5. classification performance 
       rbf_correct(rbfout, T(:,5));
   
% 6. plot decision boundary
subplot(1,2,2), rbf_db(w,P(:,1:2),4,.2), hold, plot(P(:,1),P(:,2),'.',P(list,1),P(list,2),'r.'),hold
  

%______________  Selecting Centers at Random

Xa=randn(30,2)+3; Xb=randn(30,2)+7; X_data=[Xa;Xb];  group_labels=[ones(30,1); zeros(30,1)];
subplot(1,3,1),plot(Xa(:,1),Xa(:,2),'r.',Xb(:,1),Xb(:,2),'b.'), grid

k=10; [ss,list]=sort(randn(1,60)); list=list(1:k); centers=X_data(list,:); %selecting randomly k centers
subplot(1,3,2),plot(Xa(:,1),Xa(:,2),'r.',Xb(:,1),Xb(:,2),'b.',centers(:,1),centers(:,2),'go'), grid

sigma=4; lam=1; w=rbf(X_data, centers, group_labels, sigma,1); % train the RBF with k neurons im hidden-layer and 1 neuron in out-layer

y = rbf_test(w, X_data, centers, sigma)
subplot(1,3,3),plot(y,'.-'), grid


% To evalute the RBF net, we create ne data from first group
   Xa_new=randn(50,2)+3;
   % and use them as input to the trained network
   y = rbf_test(w, Xa_new, centers, sigma)

   
%_______________ Selecting Centers via Clustering
 
 [centers, u] = FCM(X_data, 4); % 
 subplot(1,2,1),plot(X_data(:,1),X_data(:,2),'.',Xa(:,1),Xa(:,2),'r.',centers(:,1),centers(:,2),'o'),grid
 
sigma=4; lamda=1; w = rbf(X_data, centers, group_labels, sigma, lamda); % train the RBF with k neurons im hidden-layer and 1 neuron in out-layer

y = rbf_test(w, X_data, centers, sigma); subplot(1,2,2),plot(y,'.-'), grid  % apply the trained network on the same data

%________________estimating also the sigma parameter_________________
[mm,class_index]=max(u); % class index defines which vectors belong to each group
sigma_1=std(X_data(class_index==1,:)) % If we want to TRAIN the RBFs regarding the sigma  
sigma_2=std(X_data(class_index==2,:))  % we can use the grouping produced by the clustering algorithm and estimate typical radius-sizes


%______________ A simple 1-D interpolation example______________ 

t=[1:500]; signal=2*cos(2*pi*0.02*t)+3*cos(2*pi*0.015*t);

subplot(1,2,1),plot(t,signal,t(1:10:end),signal(1:10:end),'*'),grid

centers=t(1:10:end); sigma=4; lamda=1; w = rbf(t', centers', signal', sigma, lamda); %train the network using 10 centers

y=rbf_test(w,t',centers',sigma); % running the network as an interpolating device

subplot(1,2,2),plot(t,signal,t(1:10:end),signal(1:10:end),'.',t,y),grid

tt=[1:1000]; y_e=rbf_test(w,tt',centers',sigma); figure(2), subplot(1,2,1),plot(t,signal,'.',tt,y_e),grid, title('extrapolation') % extrapolation fails !!!
tt=[1:0.5:500]; y_s=rbf_test(w,tt',centers',sigma); figure(2), subplot(1,2,2),plot(t,signal,'.',tt,y_s,'.-'),grid, title('upsampling') % interpolation can offer super resolution !!!


%_________________ NonLinear_Time_Series_prediction
     
t=[1:500]; signal=2*cos(2*pi*0.02*t); % a signal reflecting activity from a simple linear system 

%____time delay embedding
figure(2),clf,subplot(4,1,1),plot(t,signal,t,signal,'r.'),subplot(4,1,2),plot(signal(1:end-1),signal(2:end)),grid,axis square
subplot(4,1,3),plot(signal(1:end-5),signal(6:end)),grid,axis square,subplot(4,1,4),plot(signal(1:end-13),signal(14:end)),grid,axis square



%_____________________________ a signal reflecting activity from a nonlinear dynamical system (Tzafestas, p.158, example 5.5 )
y(1)=0.1;y(2)=0.1;
for t=3:1000
y(t)=[0.8-0.5*exp(-[y(t-1)]^2)]*y(t-1)-[0.3+0.9*(exp(-[y(t-1)]^2))]*y(t-2)+0.1*sin(pi*y(t-1))+0.01*randn(1);
end

figure(2),clf,subplot(4,1,1),plot(y(1:250)),subplot(4,1,2),plot(y(251:500)),subplot(4,1,3),plot(y(501:750)),subplot(4,1,4),plot(y(751:1000))

figure(3),clf
subplot(1,3,1),plot(y(1:end-1),y(2:end),'.'),grid,axis square
subplot(1,3,2),plot(y(1:end-5),y(6:end),'.'),grid,axis square,subplot(1,3,3),plot(y(1:end-13),y(14:end),'.'),grid,axis square


%_____________________  MODELING USING 10 RANDOMLY SELECTED CENTERS__________
X_data=signal_to_matrix(y',3);

k=10; [ss,list]=sort(randn(1,length(X_data))); list=list(1:k); centers=X_data(list,:); %selecting randomly k centers

centers=X_data(list,1:2);
sigma=4; lamda=1; w = rbf(X_data(:,1:2), centers, X_data(:,3), sigma, lamda); %train the network using 10 centers

y_p=rbf_test(w,X_data(:,1:2),centers,sigma); % test the neural  model

figure(1),clf, subplot(1,2,1),plot(y(1:end-1),y(2:end),'.'),title('actual nonlinear system'),grid,subplot(1,2,2), plot(y_p(1:end-1),y_p(2:end),'.'),grid,title('RBF based modeled sustem') 
figure(2),clf, plot([1:200],y(3:202),[1:200],y_p(1:200)), legend('original','neural-model') 


%_______________ MODELING USING 100 RANDOMLY SELECTED CENTERS__________

k=100; [ss,list]=sort(randn(1,length(X_data))); list=list(1:k); centers=X_data(list,:); %selecting randomly k centers

centers=X_data(list,1:2);
sigma=4; lamda=1; w = rbf(X_data(:,1:2), centers, X_data(:,3), sigma, lamda); %train the network using 10 centers

y_p=rbf_test(w,X_data(:,1:2),centers,sigma); % test the neural  model

figure(3),clf, subplot(1,2,1),plot(y(1:end-1),y(2:end),'.'),title('actual nonlinear system'),grid,subplot(1,2,2), plot(y_p(1:end-1),y_p(2:end),'.'),grid,title('RBF based modeled sustem') 
figure(4),clf, plot([1:200],y(3:202),[1:200],y_p(1:200)), legend('original','neural-model') 



%_____________________  MODELING USING 10 CLUSTER-Centers__________
X_data=signal_to_matrix(y',3);
k=40; [ss,list]=sort(randn(1,length(X_data))); list=list(1:k); centers=X_data(list,:); %selecting randomly k centers

[centers, u] = FCM(X_data(:,1:2), 10);  
figure(1),subplot(1,2,1),plot(centers(:,1),centers(:,2),'.'),subplot(1,2,2),plot(X_data(:,1),X_data(:,2),'.')


sigma=4; lamda=1; w = rbf(X_data(:,1:2), centers, X_data(:,3), sigma, lamda); %train the network using 10 centers

y_p=rbf_test(w,X_data(:,1:2),centers,sigma); % test the neural  model

figure(2),clf, subplot(1,2,1),plot(y(1:end-1),y(2:end),'.'),title('actual nonlinear system'),grid,subplot(1,2,2), plot(y_p(1:end-1),y_p(2:end),'.'),grid,title('RBF based modeled sustem') 
figure(3),clf, plot([1:200],y(3:202),[1:200],y_p(1:200)), legend('original','neural-model') 



% Exercise: RBF_based chaotic prediction on the Mackay-Glass_attractor

x=[]; a=0.2; b=0.1,delay=17;

x(1:delay+1)=0.9+abs(0.01*randn(1,delay+1));
i=1, 
while i < 10000;  
    t=delay+i ;
   x=[x (1-b)*x(t)+a* (( x(t-delay)/ (1+x(t-delay).^10 )))]; 
   i=i+1;
end

subplot(2,1,1),plot(x(1:2000)),subplot(2,1,2),plot(x(1:end-9),x(10:end)),grid



