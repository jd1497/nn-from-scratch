#Clear plots
if(!is.null(dev.list())) dev.off()

# Clear console
cat("\014") 

# Clean workspace
rm(list=ls())

#helper function to print functions
print.fun=function(x){header=deparse(args(x))[1]; b=body(x);  print(gsub('function',x,header));  print(b);}

show_digit2 = function(arr784, col=gray(12:1/12),...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, xaxt='n', yaxt='n',...)
}




#install.packages('downloader')
download_mnist=function(){
  library(downloader)
  
  if(!file.exists("train-images-idx3-ubyte")) {
    if(!file.exists("train-labels-idx1-ubyte.gz")) download(url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",destfile="train-images-idx3-ubyte.gz")
    system("gunzip train-images-idx3-ubyte.gz") 
  }
  if(!file.exists("train-labels-idx1-ubyte")) {
    if(!file.exists("train-images-idx3-ubyte.gz")) download(url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",destfile="train-labels-idx1-ubyte.gz")
    system("gunzip train-labels-idx1-ubyte.gz")
  }
  if(!file.exists("t10k-images-idx3-ubyte")) {
    if(!file.exists("t10k-images-idx3-ubyte.gz")) download(url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",destfile="t10k-images-idx3-ubyte.gz")
    system("gunzip t10k-images-idx3-ubyte.gz")
  }
  if(!file.exists("t10k-labels-idx1-ubyte")) {
    if(!file.exists("t10k-labels-idx1-ubyte.gz")) download(url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",destfile="t10k-labels-idx1-ubyte.gz")
    system("gunzip t10k-labels-idx1-ubyte.gz")
  }
  print("MNIST data load complete")
}


#https://stackoverflow.com/questions/48928652/load-the-mnist-digit-recognition-dataset-with-r-and-see-any-results
load_image_file <- function(filename) {
  ret = list()
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  ret$n = readBin(f,'integer',n=1,size=4,endian='big')
  nrow = readBin(f,'integer',n=1,size=4,endian='big')
  ncol = readBin(f,'integer',n=1,size=4,endian='big')
  x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
  ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
  close(f)
  ret
}

load_label_file <- function(filename) {
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  y = readBin(f,'integer',n=n,size=1,signed=F)
  close(f)
  y
}





library(zeallot)   # defines unpacking assignment %<-%

# binary classication data 
library(MASS)
set.seed(123)
generate.gaussian.data.class3=
  function(n1=50,mu1=c(.5,.5),cov1=diag(.2,2),
           n2=40,mu2=c(1.5,1.5),cov2=diag(.1,2),
           n3=30,mu3=c(1.5,0),cov3=diag(.1,2)){
    
    x1=mvrnorm(n1,mu1,cov1)
    x2=mvrnorm(n2,mu2,cov2)
    x3=mvrnorm(n3,mu3,cov3)
    X=t(rbind(x1,x2,x3))
    y=matrix(c(rep(1,n1),rep(2,n2),rep(3,n3)),nrow=1)
    
    return(list(X,y))
  }

gen.gaussian.data.2d=function(mean.cov.list){
  data=list()
  for(params in mean.cov.list){
    n=params$n
    mu=params$mu
    cov=params$cov
    print(mu)
    data[[length(data)+1]]=mvrnorm(n,mu,cov)
  }
  X=t(do.call(rbind,data))
  y=matrix(rep(1:length(mean.cov.list),sapply(mean.cov.list,'[[',"n")),nrow=1)
  
  return(list(X,y))
}

sigmoid = function(x) 1 / (1 + exp(-x))

one.hot=function(y,classes=NULL){
  if(is.null(classes)) classes=unique(y)
  classes=sort(classes)
  hot=match(y,classes)
  t=matrix(0,nrow=length(classes),ncol=length(y))
  for(i in 1:length(hot)) t[hot[i],i]=1
  t
}
softmax=function(a){
  ea=exp(a)
  t(t(ea)/colSums(ea))
}
stable.softmax=function(a){
  a=a-max(a)
  ea=exp(a)
  t(t(ea)/colSums(ea))
}


fwd.prop=function(X,b,w){
  Z1 = b + w %*% X
  A1 = stable.softmax(Z1)
  list(Z1=Z1,A1=A1)
}

bk.prop=function(X,Y,fprop){
  M=ncol(X)
  A1=fprop$A1
  Z1=fprop$Z1
  dz=A1-Y
  db1=rowSums(dz)/M
  dw1=(dz %*% t(X))/M
  list(db=db1,dw=dw1)
}

cost=function(AL,Y){
  M=ncol(Y)
  -sum(log(AL[Y==1]))/M
}
cost.bw=function(X,Y,b,w){
  fp=fwd.prop(X,b,w)
  cost(fp$A1,Y)
}
fit=function(X,b,w){fwd.prop(X,b,w)$A1}
predict=function(X,b,w){
  s=fit(X,b,w)
  apply(s,2,which.max)
}

# X is the input data matrix with samples arranged in columns
# Y is the label matrix for X, one-hot encoded, samples arranged in columns
# w maps input vectors to output vectors. ncol(w)=nrow(X), nrow(w)=nrow(Y)
# b has an element for each output node. length(b)=nrow(Y)
softmax.fit=function(X,Y,lr=.01,max.its=100,b,w){
  trace=list()
  M=ncol(X)
  for(i in 1:max.its){
    fp=fwd.prop(X,b,w)
    bp=bk.prop(X,Y,fp)
    b = b - lr*bp$db
    w = w - lr*bp$dw
    g=sum((bp$dw)^2)
    ww=sum(w*w)
    cn=cost(fp$A1,Y)
    if(i%%100==0)print(paste(i,":",cn,ww,g))
    trace[[i]]=list(cost=cn,b=b,w=w,g=g,ts=Sys.time())
  }
  trace
}



num.gradient=function(cost,x,y,b,w,eps=1e-8){
  db=numeric(length(b))
  for(i in 1:length(b)){
    bp=bm=b
    bp[i]=bp[i]+eps
    bm[i]=bm[i]-eps
    db[i]=( cost(x,y,bp,w)-cost(x,y,bm,w))/(2*eps)
  }
  dw=numeric(length(w))
  dim(dw)=dim(w)
  for(i in 1:nrow(w)){
    for(j in 1:ncol(w)){
      wp=wm=w
      wp[i,j]=wp[i,j]+eps
      wm[i,j]=wm[i,j]-eps
      dw[i,j]=(cost(x,y,b,wp)-cost(x,y,b,wm))/(2*eps)
    }
  }
  return(list(db=db,dw=dw))
}

if(FALSE){  # synthetic data test
  #data=mlbench.spirals(75,1.5,.07)
  n1=50
  n2=40
  n3=30
  x1=mvrnorm(n1,c(.5,.5),diag(.2,2))
  x2=mvrnorm(n2,c(1.5,1.5),diag(.1,2))
  x3=mvrnorm(n3,c(1.5,0),diag(.1,2))
  X=t(rbind(x1,x2,x3))
  Y=matrix(c(rep(1,n1),rep(2,n2),rep(3,n3)),nrow=1)
  plot(X[1,],X[2,],pch=1,col=Y,lwd=2,cex=1)
  #Y=one.hot(t(Y))
  #plot(X[1,],X[2,],pch=1,col=apply(Y,2,which.max)+2,lwd=2,cex=1)
  
  n = nrow(X)  # number of input nodes
  n.out = nrow(Y)
  M = ncol(X)  # number of samples
  
  
  #b2 = runif(n.out,-.1,.1)
  #w2 = matrix(rnorm(1*nh,0,.1),nrow=n.out,ncol=nh)
  
  #data$color=as.integer(data$classes)+2
  grid=expand.grid(X1 = seq(min(X[1,])-.25,max(X[1,])+.25,length.out = 101),
                   X2 = seq(min(X[2,])-.25,max(X[2,])+.25,length.out = 101))
  grid=t(data.matrix(grid))
  
  # initialize unknowns - these should be collected in list called 'model'
  #nh = 10
  b0 = runif(n.out,-.1,.1)
  w0 = matrix(rnorm(n*n.out,0,.1),nrow=n.out,ncol=n)
  # it would be nice to add split cost/accuracy plot
  
  lr=.1
  c(cost,b,w,g) %<-% softmax.fit(X,Y,lr,max.its=2000,b0,w0)
  # predict returns index of predicted class: 1,2,...n
  errors=predict(X,b,w)!=apply(Y,2,which.max)
  print(sum(errors))
  pgrid=predict(grid,b,w)
  plot(X[1,],X[2,],pch=1,col=apply(Y,2,which.max)+2,lwd=2,cex=1)
  points(grid[1,],grid[2,],col=pgrid+3,cex=.4)
  wrong=which(errors)
  for(i in wrong) points(X[1,i],X[2,i],lwd=2,col="red",cex=2)
  if(rate < 1) lr=lr/2.
}


if(FALSE){  # try running mnist
  source("softreg_v2.R")
  mnist=load.mnist.data()
  train.size=5000
  train.set=sample(1:nrow(mnist$train_images),train.size)
  X=t(mnist$train_images[train.set,])/255.
  # for testing
  x1=mnist$train_images[1,]/255
  Y=one.hot(mnist$train_labels[train.set])
  
  # ncol(w)=nrow(X), nrow(w)=nrow(Y)
  
  
  lr=.5
  max.its=2000
  b=numeric(nrow(Y)) 
  w=matrix(0,nrow=nrow(Y),ncol=nrow(X))
  w=matrix(rnorm(nrow(X)*nrow(Y),0,1),nrow=nrow(Y))
  # the orginal train function used global b,w and M
  #M= ncol(X)
  #foo1=train(X,Y,lr=lr,max.its=max.its)
  history=softmax.fit(X,Y,lr=lr,max.its=max.its,b,w)
  c(final.cost,b,w,g) %<-% history[[length(history)]]
  yhat=predict(X,foo$b,foo$w)
  table(yhat-1==mnist$train_labels[train.set])
}

if(FALSE){  # check derivs - b is now a vector
  # computes all derivatives - only useful if a small number of parameters
  
  ng=num.gradient(cost.bw,X,Y,b,w)
  grad=b.prop(X,Y,b,w)
  
  grad$db/ncol(X)
  grad$dw/ncol(X)
  
  cols=1:10
  ng=num.gradient(cost.bw,X[,cols,drop=FALSE],Y[,cols,drop=FALSE],b,w)
  grad=b.prop(X[,cols,drop=FALSE],Y[,cols,drop=FALSE],b,w)
  ng$db
  grad$db1
  dim(ng$dw)
  # pick 10 random pairs and check w-derivs
  N=10
  I=matrix(c(sample(1:nrow(w),N,replace=TRUE),sample(1:ncol(w),N,replace=TRUE)),nrow=N)
  ng$dw[I]
  grad$dw[I]
}

accuracy=function(X,Y,b,w){
  pred=predict(X,b,w)
  sum(pred == Y)/length(Y)
}
cost.history=function(history){ sapply(history,function(el){el$cost}) }
wgt.history=function(history){ 
  b=sapply(history,function(el){el$b}) 
  w=sapply(history,function(el){el$w})
  matrix(c(b,w),nrow=length(b))
}
perf.history=function(xin,y,history){
  perf=function(el){accuracy(xin,y,el$b,el$w)}
  sapply(history,perf)
}
