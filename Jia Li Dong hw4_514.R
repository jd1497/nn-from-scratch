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


fwd.prop=function(X,b,w){
  onemat = matrix(1,ncol=ncol(X))
  Z = (b %*% onemat) + (w %*% X)
  return (t(t(exp(Z))/colSums(exp(Z))))
}

bk.prop=function(X,Y,fprop){
  
  m= ncol(X)
  ones = matrix(1,nrow=m)
  dz= ((fprop)-Y)
  dw = (1/m)*(dz%*%t(X))
  db = (1/m)*rowSums(dz)
  return(list(db=db,dw=dw))
}


cost=function(X,Y,b,w){
  #one hot the y first before having it be t
  t = Y
  m= ncol(fwd.prop(X,b,w))
  cost = (-1/m)*sum(t*log(fwd.prop(X,b,w)))
  return (cost)
}



predict=function(X,b,w){
  Hmat = fwd.prop(X,b,w)
  pred = NULL
  for (i in 1:ncol(Hmat)){
    maxdeter = which(Hmat[,i]==max(Hmat[,i]))
    pred=append(pred,maxdeter)}
  return(pred)
}

# X is the input data matrix with samples arranged in columns
# Y is the label matrix for X, one-hot encoded, samples arranged in columns
# w maps input vectors to output vectors. ncol(w)=nrow(X), nrow(w)=nrow(Y)
# b has an element for each output node. length(b)=nrow(Y)
softmax.fit=function(X,Y,lr=.01,max.its=100,b,w){
  trace = list()
  for (i in 1:max.its){
    E = fwd.prop(X,b,w) - Y
    c(db,dw) %<-% num.gradient(cost,X,Y,b,w)
    costup=cost(X,Y,b,w)
    #if(is.nan(cost)) print(paste(b,w))
    trace[[i]]=list(cost=costup,b=b,w=w,db=db,dw=dw,time=Sys.time())
    b = b - (lr * db)
    w = w- (lr * dw)
  }
  return(trace)
}



num.gradient=function(cost,x,y,b,w,eps=1e-8){
  dwi=numeric(length(w))
  db=numeric(length(b))
  for (i in 1:length(b)){
    bp=bm=b
    bp[i]=bp[i]+eps
    bm[i]=bm[i]-eps
    db[i]=( cost(x,y,bp,w)-cost(x,y,bm,w))/(2*eps)
  }
  for (i in 1:length(w)){
    wp=wm=w
    wp[i]=wp[i]+eps
    wm[i]=wm[i]-eps
    dwi[i]=( cost(x,y,b,wp)-cost(x,y,b,wm))/(2*eps)
  } 
  dw = matrix(dwi,ncol=ncol(w),nrow=nrow(w))
  return(list(db=db,dw=dw))
}

accuracy=function(X,Y,b,w){
  pred=predict(X,b,w)
  sum(pred == Y)/length(Y)
}

plot.cost.history=function(history){
  plot(unlist(lapply(history,'[',"cost")),main="Cost History",ylab="Cost")
}
plot.grad.dw.history=function(history){
  grad.norm=function(el){ sqrt((el$db)^2+sum(el$dw*el$dw)) }
  plot(unlist(lapply(history,'[',"dw")),main="Gradient of w History",ylab="Gradient of w")#(sapply(history,grad.norm))
}
plot.grad.db.history=function(history){
  grad.norm=function(el){ sqrt((el$db)^2+sum(el$dw*el$dw)) }
  plot(unlist(lapply(history,'[',"db")),main="Gradient of b History",ylab="Gradient of b")#(sapply(history,grad.norm))
}
plot.wgt.w.history=function(history){
  wgt.norm=function(el){ sqrt((el$b)^2+sum(el$dw*el$w)) }
  plot(unlist(lapply(history,'[',"w")))#(sapply(history,grad.norm))
}
plot.perf.history=function(xin,y,history){
  perf=function(el){accuracy(xin,y,el$b,el$w)}
  plot(sapply(history,perf),main="Performance History",ylab="Performance Accuracy")
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
