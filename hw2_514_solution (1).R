#Clear plots
if(!is.null(dev.list())) dev.off()

# Clear console
cat("\014") 

# Clean workspace
rm(list=ls())

#helper function to print functions
print.fun=function(x){header=deparse(args(x))[1]; b=body(x);  print(gsub('function',x,header));  print(b);}


# randomly place points in the box xlim=c(0,1),ylim=c(0,3)
# The two classes have a minimum distance of 2*gamma
# If gamma < 0, then the classes are not separable
perceptron.box.data=function(n,gamma=.25,seed=NULL){
  require(zeallot)
  if(!is.null(seed)) set.seed(seed)
  data=matrix(0,nrow=n,ncol=3)
  # the discriminant mid-point line runs from (0,1) to (1,2) (slope 1)
  discriminant=function(x,y){(1+x-y)/sqrt(2)}
  m=0
  while(m < n){
    x=runif(1,0,1)
    y=runif(1,0,3)
    d=discriminant(x,y)
    d1=d >= gamma
    d2=d <= -gamma
    if(d1 & !d2){
      m=m+1
      data[m,] %<-% c(x,y,+1)
    }else if(d2 & !d1){
      m=m+1
      data[m,] %<-% c(x,y,-1)
    }else if(d1 & d2){
      m=m+1
      data[m,] %<-% c(x,y,sample(c(-1,1),1))
    }
  }
  data
}


plot.perceptron.box.data=function(data,title='perceptron'){
  col=data[,3]+3
  col=c("red","blue")[col/2]
  
  plot(data[,1],data[,2],type="p",col=col,pch=16,main=title,ylim=c(min(data[,2]),max(data[,2])),xlim=c(min(data[,1]),max(data[,1])))
}


perceptron.train=function(x,y,epoch=100){
  w=rep(0,ncol(x))
  b=0
  mistakes=0;
  
  for (i in 1:epoch){
    epoch.mistakes=mistakes;
    for (j in 1:dim(x)[1]){
      a=sum(w*x[j,])+b    
      if (y[j]*a<=0){
        w=w+y[j]*x[j,]
        b=b+y[j]
        mistakes=mistakes+1
      }
    }
    if (epoch.mistakes==mistakes) break;
  }
  return(list(w=w,b=b,mistakes=mistakes))
}

predict.perceptron=function(w,b,x){
  a=x%*%w+b   
  return(2*as.integer(a>=0)-1)
}

euclidean.norm = function(x){
  sqrt(sum(x * x))
}


run.experiments=function(data,num_trials=1,epochs=100){
  
  results=vector(mode="list")
  for( i in 1:num_trials){
    #randomly shuffle data
    data_trial=data[sample(nrow(data)),]
    #print(epochs)
    results[[i]]=perceptron.train(data_trial[,1:2],data_trial[,3],epochs)
  }
  
  return(results)
}



avg.perceptron.train=function(x,y,iter=100){
  w=rep(0,ncol(x))
  b=0
  
  u=rep(0,ncol(x))
  v=0
  c=1
  
  mistakes=0;
  for (i in 1:iter){
    epoch.mistakes=mistakes;
    for (j in 1:dim(x)[1]){
      a=sum(w*x[j,])+b    
      if (y[j]*a<=0){
        w=w+y[j]*x[j,]
        b=b+y[j]
        u=u+y[j]*c*x[j,]
        v=v+y[j]*c
        mistakes=mistakes+1
      }
      c=c+1
    }
    if (epoch.mistakes==mistakes) break;
  }
  return(list(w=w-1/c*u,b=b-v/c,mistakes=mistakes))
}





fisher=function(x,y){
  x1=x[y<0,]
  x2=x[y>=0,]
  n1=nrow(x1)
  n2=nrow(x2)
  m1=colMeans(x1)
  m2=colMeans(x2)
  m = (n1*m1+n2*m2)/(n1+n2)
  
  z = rbind(sweep(x1,2,m1),sweep(x2,2,m2))
  Sw = t(z) %*% z
  w=solve(Sw,m2-m1)
  
  return(list(w=w,m=m))
}

fisher.discriminant=function(x,w,m){
  if(is.matrix(x)){
    as.integer(sweep(x,2,m) %*% matrix(w,nrow=length(w)) > 0)
  }else{
    as.integer(sum(w*(x-m)) > 0)
  }
}

fisher.2d.line=function(w,m){
  a=0
  b=0
  if(length(w)!=2) stop("Fisher fit is not 2d")
  wm = sum(w*m);
  
  a=wm/w[2];
  b=-w[1]/w[2];  # not sure if the a, b do anything
  return(list(a=a,b=b))
}

sign=function(a) {2*as.integer(a>=0)-1}

compute_vote=function(wgts,cs,bs,x)
{
  a=0
  for (i in length(wgts)){
    a=a+cs[[i]]*sign(sum(wgts[[i]]*x)+bs[[i]])
  }
  a=sign(a)  
  
}

# voted.perceptron.train=function(x,y,iter=200){
#   w=rep(0,ncol(x))
#   wgts=vector(mode='list')
#   cs=vector(mode='list')
#   bs=vector(mode='list')
#   b=0
#   
#   u=rep(0,ncol(x))
#   v=0
#   c=1
#   wgts[[1]]=w
#   cs[[1]]=c
#   bs[[1]]=b
#   mistakes=0;
#   
#   for (i in 1:iter){
#     
#     for (j in 1:nrow(x)){
#       #a=sum(w*x[j,])+b    ;
#       a=compute_vote(wgts,cs,bs,x[j,]);
#       
#       if (y[j]*a<=0){
#         
#         mistakes=mistakes+1
#         wgts[[mistakes]]=w
#         cs[[mistakes]]=c
#         bs[[mistakes]]=b
#         
#         w=w+y[j]*x[j,]
#         b=b+y[j]
#         
#         c=1
#       }
#       else {c=c+1}
#     }
#   }
#   return(list(wgts=wgts,cs=cs,bs=bs,mistakes=mistakes))
# }
# 
# predict.voted=function(wgts,cs,bs,x){
#   a=0
#   aa=list();
#   
#   for (j in 1:nrow(x)){
#     aa[[j]]=compute_vote(wgts,cs,bs,x[j,])
#   }
#   return(unlist(aa))
#   
# }


sgn=function(x){2*as.integer(x>=0)-1}
freund.voted.perceptron.train=function(x,y,epochs=200){
  history=list()
  
  w=rep(0,ncol(x))
  b=0
  c=1
  mistakes=0;
  
  for (i in 1:epochs){
    start.mistakes=mistakes;
    for (j in 1:nrow(x)){
      a=sgn(b+sum(w*x[j,]))
      
      if (y[j]*a<=0){
        mistakes=mistakes+1
        history[[length(history)+1]]=list(b=b,w=w,c=c)
        w=w+y[j]*x[j,]
        b=b+y[j]
        c=1
      }else{
        c=c+1
      }
    }
    if (start.mistakes==mistakes) break;
  }
  history[[length(history)+1]]=list(b=b,w=w,c=c)
  print(i)
  list(history=history,mistakes=mistakes)
}
compute.vote=function(history,x){
  ifelse(length(history)==0,0, sum(sapply(history,function(h){h$c*sgn(h$b+sum(h$w*x))})))
}

#this function will apply the wgts, cs and bs to the entire data set
predict.voted=function(history,x){
  y=numeric(nrow(x))
  
  for (j in 1:nrow(x)){
    #print(j)
    y[j]=compute.vote(history,x[j,])
  }
  sgn(y)
}

margins=function(b,w,x){
  ww=sum(w*w)
  -x[,ncol(x)]*(b+x[,-ncol(x)]%*%w)/sqrt(ww)
}
margin=function(b,w,x){
  ww=sum(w*w)
  distances=margins(b,w,x)
  if(all(distances>=0)){
    print("all positive")
    min(distances)
  }else{
    sum(distances[distances<0])
  }
}