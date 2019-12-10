#Clear plots
if(!is.null(dev.list())) dev.off()

# Clear console
cat("\014") 

# Clean workspace
rm(list=ls())

#helper function to print functions
print.fun=function(x){header=deparse(args(x))[1]; b=body(x);  print(gsub('function',x,header));  print(b);}


#source("./nn_fns.R")
require(zeallot)
require(plyr)
require(mvtnorm)

sigmoid = function(x) 1 / (1 + exp(-x))

# provide means and sd for each component
gaussian.mixture.1d=function(n1,m1,s1,n2,m2,s2){
  data=matrix(0,nrow=n1+n2,ncol=2)
  data[(1:n1),1]=rnorm(n1,m1,s1)
  data[(1:n1),2]=1
  data[(n1+1):nrow(data),1]=rnorm(n2,m2,s2)
  data[(n1+1):nrow(data),2]=2
  data[sample(1:nrow(data),nrow(data)),]
}

generate.1d.dataset=function(n1=30,m1=-1,s1=1,n2=70,m2=2,s2=1){
  #n1=30; m1=-1; s1=1; n2=70; m2=2; s2=1
  data=gaussian.mixture.1d(n1,m1,s1,n2,m2,s2)
  x=matrix(data[,1],nrow=nrow(data))
  y=as.integer(data[,2]==max(data[,2]))
  list(x=x,y=y)
}


gaussian.mixture.2d=function(n1,m1,c1,n2,m2,c2){
  x1=mvrnorm(n1,m1,c1)
  x1=cbind(x1,1)
  x2=mvrnorm(n2,m2,c2)
  x2=cbind(x2,2)
  x =rbind(x1,x2)
  x[sample(1:nrow(x),nrow(x)),]
}

generate.2d.dataset=function(n1=30,m1=c(1,1),c1=diag(c(1,1)),n2=70,m2=c(3,3),c2=c(1,1)){
  n1=30; m1=c(1,1); c1=diag(c(1,1))
  n2=70; m2=c(3,3); c2=diag(c(3,3))
  data=gaussian.mixture.2d(n1,m1,c1,n2,m2,c2)
  x=data[,-ncol(data)]
  y=as.integer(data[,ncol(data)]==max(data[,ncol(data)]))
  list(x=x,y=y)
}


one.hot=function(y,classes=NULL){
  if(is.null(classes)) classes=unique(y)
  classes=sort(classes)
  hot=match(y,classes)
  t=matrix(0,nrow=length(classes),ncol=length(y))
  for(i in 1:length(hot)) t[hot[i],i]=1
  t
}

if(FALSE){  # look at baye's rule given known dist
  # p(x|c2) = dnorm(x,m2,s2)
  ratio = function(x,n1,m1,s1,n2,m2,s2){
    dnorm(x,m2,s2)*n2/(dnorm(x,m1,s1)*n1)
  }
  ff=function(x){ratio(x,n1,m1,s1,n2,m2,s2)-0.5}
  grid=seq(-2,.5,length.out = 100)
  plot(grid,ratio(grid,n1,m1,s1,n2,m2,s2))
  uniroot(ff,c(-2,2))
}


# this function is not used by logistic code - it is used to look as cost as a function of b,w
cost=function(x,y,b,w,neg.logll=TRUE){
  yhat=f.prop(x,b,w)
  m = length(yhat)
  if(neg.logll){
    #cost=(-1/m)*sum(y*log(yhat)+(1-y)*log(1-yhat))
    cost=(-1/m)*(sum(log(yhat[y==1]))+sum(log(1-yhat[y==0])))
  }else{
    e = matrix(y - yhat,ncol=1)
    cost=(1/(2*m))*sum(e^2)
  }
  cost
}
f.prop=function(x,b,w){
  sigmoid(b+ w %*% x)
}

b.prop=function(x,y,yhat,neg.logll){
  m = length(yhat)
  e = matrix(y - yhat,ncol=1)
  if(neg.logll){
    #cost=(-1/m)*sum(y*log(yhat)+(1-y)*log(1-yhat))
    cost=(-1/m)*(sum(log(yhat[y==1]))+sum(log(1-yhat[y==0])))
    db=(-1/m)*sum(e)
    dw=(-1/m)*(x %*% e)
  }else{
    cost=(1/(2*m))*sum(e^2)
    r = e*t(yhat*(1-yhat))
    #print(dim(r))
    db=(-1/m)*sum(r)
    dw=(-1/m)*(x %*% r)
  }
  list(cost=cost,db=db,dw=t(dw))
}

# mike added y argument on 2/10
num.gradient=function(cost,x,y,b,w,eps=1e-8,neg.logll=TRUE){
  dw=numeric(length(w))
  db=( cost(x,y,(b+eps),w,neg.logll)-cost(x,y,(b-eps),w,neg.logll))/(2*eps)
  for (i in 1:length(w)){
    wp=wm=w
    wp[i]=wp[i]+eps
    wm[i]=wm[i]-eps
    dw[i]=( cost(x,y,b,wp,neg.logll)-cost(x,y,b,wm,neg.logll))/(2*eps)
  } 
  return(list(db=db,dw=dw))
}

log.fit = function(x,y,neg.logll,eta,max.its,tol,b.init=0,w.init=0){
  trace = list()
  b = b.init
  w = matrix(w.init,nrow=1,ncol=nrow(x))
  for (i in 1:max.its) {
    yhat = f.prop(x,b,w)
    c(cost,db,dw) %<-% b.prop(x,y,yhat,neg.logll)
    if(is.nan(cost)) print(paste(b,w))
    trace[[i]]=list(cost=cost,b=b,w=w,db=db,dw=dw)
    w = w - eta * dw
    b = b - eta * db
  }
  trace
}
predict=function(xin,b,w){
  x=t(xin)
  yhat=f.prop(x,b,w)
  as.integer(yhat >= 0.5)
}
accuracy=function(xin,y,b,w){
  pred=predict(xin,b,w)
  sum(pred == y)/length(y)
}
logistic.2d.line=function(b,w){
  c(-b/w[2],-w[1]/w[2])
} 
plot.cost.history=function(history){
  plot(unlist(lapply(history,'[',"cost")))
}
plot.grad.history=function(history){
  grad.norm=function(el){ sqrt((el$db)^2+sum(el$dw*el$dw)) }
  plot(sapply(history,grad.norm))
}
plot.perf.history=function(xin,y,history){
  perf=function(el){accuracy(xin,y,el$b,el$w)}
  plot(sapply(history,perf))
}
cost.history=function(history){ sapply(history,function(el){el$cost}) }
wgt.history=function(history){ 
  b=sapply(history,function(el){el$b}) 
  w=sapply(history,function(el){el$w})
  matrix(c(b,w),nrow=length(b))
}
grad.history=function(history){
  db=sapply(history,function(el){el$db}) 
  dw=sapply(history,function(el){el$dw})
  sqrt(db*db+sum(dw*dw))
}
perf.history=function(xin,y,history){
  perf=function(el){accuracy(xin,y,el$b,el$w)}
  sapply(history,perf)
}
# don't like that x,y,neg.logll are taken from global env
cost.func=function(b,w){
  cost(x,y,b,w,neg.logll)
}
grad.func=function(b,w){
  yhat=f.prop(x,b,w)
  c(cost,db,dw) %<-% b.prop(x,y,yhat,neg.logll)
  sqrt(db*db + sum(dw*dw))
}
eval.grid=function(history,func,nx=50,ny=50,b.margin=2,w.margin=2){
  b0 = history[[1]]$b
  w0 = history[[1]]$w
  bf = history[[length(history)]]$b
  wf = history[[length(history)]]$w
  bmin = min(b0,bf)-b.margin
  bmax = max(b0,bf)+b.margin
  wmin = min(w0,wf)-w.margin
  wmax = max(w0,wf)+w.margin
  b.seq = seq(bmin,bmax,length.out = nx)
  w.seq = seq(wmin,wmax,length.out = ny)
  value = matrix(0,nrow=length(b.seq),ncol=length(w.seq))
  for(i in 1:length(b.seq)){
    for(j in 1:length(w.seq)){
      value[i,j]=func(b.seq[i],w.seq[j])
    }
  }
  list(bv=b.seq,wv=w.seq,value=value)
}

run.1d.test = function(neg.logll=TRUE,eta=.2,max.its=200){
  # 1d, 
  c(xin,y) %<-% generate.1d.dataset()
  x=t(xin)
  history = do.fit(x,y,neg.logll=neg.logll,eta=eta,max.its=max.its,tol)
  c(final.cost,b,w,db,dw) %<-% history[[length(history)]]
  print(accuracy(xin,y,b,w))
  list(xin=xin,y=y,history=history)
}

run.2d.test = function(neg.logll=TRUE,eta=.2,max.its=200){
  # 2d, 
  c(xin,y) %<-% generate.2d.dataset()
  x=t(xin)
  history = do.fit(x,y,neg.logll=neg.logll,eta=eta,max.its=max.its,tol)
  c(final.cost,b,w,db,dw) %<-% history[[length(history)]]
  print(accuracy(xin,y,b,w))
  list(xin=xin,y=y,history=history)
}

if(FALSE){  # try test code
  test1 = run.1d.test()
  plot.cost.history(test1$history)
  plot.grad.history(test1$history)
  plot.perf.history(test1$xin,test1$y,test1$history)
}


if(FALSE){  # try test code
  test1 = run.2d.test()
  plot.cost.history(test1$history)
  plot.grad.history(test1$history)
  plot.perf.history(test1$xin,test1$y,test1$history)
  
  b = test1$history[[length(test1$history)]]$b
  w = test1$history[[length(test1$history)]]$w
  plot(test1$xin,col=c("green","blue")[test1$y+1],pch=16)
  abline(logistic.2d.line(b,w),col="red",lwd=2)
}


if(FALSE){  # code to look at cost function contours
  nx=50; ny=50
  bv=seq(-10,10,length.out=100)
  wv=seq(-10,10,length.out=100)
  costs=matrix(0,nrow=length(bv),ncol=length(wv))
  for(i in 1:length(bv)){
    for(j in 1:length(wv)){
      costs[i,j]=cost(x,y,bv[i],wv[j],neg.logll)
    }
  }
  contour(bv,wv,costs,nlevels=40)
}

