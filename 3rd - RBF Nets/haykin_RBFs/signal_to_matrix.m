function    u = signal_to_matrix(x,l)

% u = signal_to_matrix(x,l)
% signal-vector [x(1), x(2),.....x(T)]
%                                            
%    is converted  to array: U=[x(1),x(2),...x(l); 
%                               x(2),x(3),...x(l+1);                    
%                               ..............,,;                                                                  
%                               .............x(T)]                
    

[n ,T]=size(x);
   
   
      
    u=[];
    
    for i=l:T ;
      u=[u;  x(i-l+1 :i )];
     end  
    

