#Clear plots
if(!is.null(dev.list())) dev.off()

# Clear console
cat("\014") 

# Clean workspace
rm(list=ls())

#helper function to print functions
print.fun=function(x){header=deparse(args(x))[1]; b=body(x);  print(gsub('function',x,header));  print(b);}

# one-hidden layer logistic net
# n inputs, nh hidden nodes (one output)
# only changes to regression code
# - change cost function
# - add activation to output node (in fwd.prop)

#source("nn_fns.R")
#source("../lib/data_sources.R")
library(zeallot)   # defines unpacking assignment %<-%


# see: https://www.aaai.org/Papers/AAAI/1997/AAAI97-084.pdf
# lawrence, giles: Lessons in Neural Network Training
data.lawrence.giles=function(seed=NULL){
  if(!is.null(seed)) set.seed(seed)
  # break
  X1 = seq(0,pi,length.out = 7)[c(-1,-7)]
  X2 = seq(pi,2*pi,length.out=15)
  Y1 = -cos(X1)+runif(length(X1),-.25,.25)
  Y2 = cos(3*(X2-pi))+runif(length(X2),-.25,.25)
  X = c(X1,X2)
  dim(X)=c(1,length(X))
  Y = c(Y1,Y2)
  dim(Y)=c(1,length(Y))
  X.grid=seq(0,2*pi,length.out = 99)
  Y.grid=c(-cos(seq(0,pi,length.out=50))[-1],cos(3*(seq(pi,2*pi,length.out = 50)-pi)))
  X=X/(2*pi)
  X.grid=X.grid/(2*pi)
  list(X=X,Y=Y,X.grid=X.grid,Y.grid=Y.grid)
}
plot.lawrence.giles=function(lg.data){
  plot(lg.data$X.grid,lg.data$Y.grid)
  points(lg.data$X,lg.data$Y,col=3,lwd=3)
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


# binary classication data 
library(mlbench)

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
  test=NULL
  for (i in 1:nrow(a)){
    test=append(test,max(a[i,]))
    }
  a=a-test
  #a=a-max(a) #need max
  ea=exp(a)
  t(t(ea)/colSums(ea))
}

cost.squared.error=function(x,y,b1,w1,b2,w2,f,g){
  onemat = matrix(1,ncol=ncol(x))
  Z = (b %*% onemat) + (w %*% x)
  yhat=fwd.prop(f,g,x,b1,w1,b2,w2)
  m = length(yhat)
  e = matrix(y - yhat,ncol=1)
  cost=(1/(2*m))*sum(e^2)
  return(cost)
}
cost.negll=function(x,y,b1,w1,b2,w2,f,g){
  onemat = matrix(1,ncol=ncol(x))
  Z = (b %*% onemat) + (w %*% x)
  yhat=fwd.prop(f,g,x,b1,w1,b2,w2)
  m = length(yhat)
  #cost=(-1/m)*sum(y*log(yhat)+(1-y)*log(1-yhat))
  cost=(-1/m)*(sum(log(yhat[y==1]))+sum(log(1-yhat[y==0])))
  return(cost)
}
cost.cross.entropy=function(X,y,b1,w1,b2,w2,f,g){
  #one hot the y first before having it be t
  onemat = matrix(1,ncol=ncol(X))
  Z = (b %*% onemat) + (w %*% X)
  #fwdprop= (t(t(exp(Z))/colSums(exp(Z))))
  fwdprop=fwd.prop(f,g,x,b1,w1,b2,w2)
  t = one.hot(t(y))
  m= ncol(fwdprop)
  cost = (-1/m)*sum(t*log(fwdprop))
  return (cost)
}

identity = function(x){x}
sigmoid = function(x) {1 / (1 + exp(-x))}

relu = function(x) {ifelse(x >= 0, x, 0)}
tanh = function(x) {tanh(x)}

dsigmoid=function(x){sigmoid(x)*(1-sigmoid(x))}
drelu = function(x) {ifelse(x >= 0, 1, 0)}
dtanh = function(x){1-(tanh(x)^2)}


chunker.cl=function(M,chunk.size){
  chunk=0
  defect= M %% chunk.size
  num.chunks=floor((M+chunk.size-1)/chunk.size)
  # this is very wasteful if chunk.size evenly divides M
  chunks=rep(chunk.size,num.chunks)
  if(sum(chunks)>M) chunks[num.chunks]=M-sum(chunks[-num.chunks])
  sum.chunks=0 # placeholder
  sequence=0   # placeholder
  
  function(){
    if(chunk==0){
      chunks<<-chunks[sample(1:num.chunks,num.chunks,replace=FALSE)]  # reorder chunks
      sum.chunks<<-cumsum(chunks)
      sequence<<-sample(1:M,M,replace=FALSE)
    }
    chunk<<-chunk+1
    if(chunk > num.chunks) {
      chunk<<-0; return(NULL)
    }else{
      start=ifelse(chunk==1,1,sum.chunks[chunk-1]+1)
      sequence[start:sum.chunks[chunk]]
    }
  }
}

if(FALSE){ # example, how to use chunker.cl
  m=32   # number of samples
  chunker=chunker.cl(m,12)  # 12=mini-batch size
  test.chunker=c()
  while(TRUE){
    samples=chunker()
    if(is.null(samples)) break;
    test.chunker=c(test.chunker,samples)
    print(samples)
  }
  all(sort(test.chunker)==(1:m))
}

# alternative to chunker - nicer sematics

seq.generator.cl=function(M,seq.size){
  num.chunks=floor((M+seq.size-1)/seq.size)
  sequence=0     # placeholder
  chunk.start=0  # placeholder
  defect.size= M %% seq.size
  defect.start=0 # placeholder
  seq.not.done=function(){
    response = (chunk.start <= M)
    if(response==FALSE){
      chunk.start <<- 0
    }
    response
  }
  next.seq=function(){
    if(chunk.start==0){ # start new sequence
      chunk.start <<- 1
      sequence <<- sample(1:M,M,replace=FALSE)
      if(defect.size != 0) {
        defect.start <<- (sample(num.chunks,1)-1)*seq.size+1
      }
    }
    if(defect.size==0){
      seq.chunk=sequence[chunk.start:(chunk.start+seq.size-1)]
      chunk.start <<- chunk.start+seq.size
    }else{
      size=ifelse(chunk.start==defect.start,defect.size,seq.size)
      seq.chunk=sequence[chunk.start:(chunk.start+size-1)]
      chunk.start <<- chunk.start+size
    }
    seq.chunk
  }
  list(seq.not.done=seq.not.done,next.seq=next.seq)
}
if(FALSE){
  require(zeallot)
  M=12
  batch.size=7
  c(seq.not.done,next.seq) %<-% seq.generator.cl(M,batch.size)
  for(epochs in 1:3){
    sequence=c()
    while(seq.not.done()){
      sequence=c(sequence,next.seq())
      print(sequence)
    }
    print(all(1:M == sort(sequence)))
  }
}


init.wgts=function(n.in,n.hid,n.out){
  b1 = runif(n.hid,-.1,.1)
  w1 = matrix(rnorm(n.in*n.hid,0,.1),nrow=n.hid,ncol=n.in)
  b2 = runif(n.out,-.1,.1)
  w2 = matrix(rnorm(n.out*n.hid,0,.1),nrow=n.out,ncol=n.hid)
  list(b1=b1,w1=w1,b2=b2,w2=w2)
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

nnet1.fit=function(x,y,b1,w1,b2,w2,f,g,df,cost,lr=.01,max.its=100){
  trace=list()
  M=ncol(X)
  for(i in 1:max.its){
    fp=fwd.prop(f,g,x,b1,w1,b2,w2)
    bp=bk.prop(x,y,b1,w1,b2,w2,f,g,df)
    b1 = b1 - lr*bp$db1
    w1 = w1 - lr*bp$dw1
    b2 = b2 - lr*bp$db2
    w2 = w2 - lr*bp$dw2
    grad=sum((bp$dw1)^2)
    ww=sum(w1*w1)
    if (cost=="cost.cross.entropy"){cn=cost.cross.entropy(x,y,b1,w1,b2,w2,f,g)}
    if (cost=="cost.negll"){cn=cost.negll(x,y,b1,w1,b2,w2,f,g)}
    if (cost=="cost.squared.error"){cn=cost.squared.error(x,y,b1,w1,b2,w2,f,g)}
    if(i%%100==0)print(paste(i,":",cn,ww,grad))
    trace[[i]]=list(cost=cn,b1=b1,w1=w1,b2=b2,w2=w2,grad=grad,ts=Sys.time())
  }
  trace
}

nnet1.fit.batch=function(x,y,b1,w1,b2,w2,f,g,df,cost,batch.size,lr=.01,epoch){
  trace=list()
  M=ncol(x)
  for(i in 1:epoch){
    repitition=rep(test.chunker,length.out=ncol(x))
    test=split(x,rep(repitition,each=nrow(x)))
    for (k in 1:batch.size){
      
      kk=paste(k,sep="")
      minx = matrix(test$kk,nrow=nrow(X))
      fp=fwd.prop(f,g,minx,b1,w1,b2,w2)
      bp=bk.prop(minx,y,b1,w1,b2,w2,f,g,df)
      b1 = b1 - lr*bp$db1
      w1 = w1 - lr*bp$dw1
      b2 = b2 - lr*bp$db2
      w2 = w2 - lr*bp$dw2
      grad=sum((bp$dw1)^2)
      ww=sum(w1*w1)
      if (cost=="cost.cross.entropy"){cn=cost.cross.entropy(minx,y,b1,w1,b2,w2,f,g)}
      if (cost=="cost.negll"){cn=cost.negll(minx,y,b1,w1,b2,w2,f,g)}
      if (cost=="cost.squared.error"){cn=cost.squared.error(minx,y,b1,w1,b2,w2,f,g)}
      if(i%%100==0)print(paste(i,":",cn,ww,grad))}
    trace[[i]]=list(cost=cn,b1=b1,w1=w1,b2=b2,w2=w2,grad=grad,ts=Sys.time())
  }
  trace
}

bk.prop=function(x,y,b1,w1,b2,w2,f,g,df){
  m= ncol(x)
  a = x
  onemat = matrix(1,ncol=ncol(a))
  if (f=="sigmoid"){a1 = sigmoid((b1 %*% onemat) + (w1 %*% a))}
  if (f=="relu"){a1 = relu((b1 %*% onemat) + (w1 %*% a))}
  if (f=="tanh"){a1 = tanh((b1 %*% onemat) + (w1 %*% a))}
  
  fprop = fwd.prop(f,g,x,b1,w1,b2,w2)
  t=one.hot(t(y))
  dz=fprop-t
  db2 = rowSums(dz) #*(1/m)
  dw2 = (rowSums(dz)%*%t(a1))#*(1/m)
  if (df=="dsigmoid"){d = dsigmoid((b1 %*% onemat) + (w1 %*% a))}
  if (df=="drelu"){d = drelu((b1 %*% onemat) + (w1 %*% a))}
  if (df=="dtanh"){d = dtanh((b1 %*% onemat) + (w1 %*% a))}
  dz = d * (t(w2)%*%dz)
  db1 = rowSums(dz)
  dw1 = (rowSums(dz)%*%t(a))
  return(db1=db1,dw1=dw1,db2=db2,dw2=dw2)
}

fwd.prop=function(f,g,x,b1,w1,b2,w2){
  a0=x
  onemat = matrix(1,ncol=ncol(a0))
  Z = (b1 %*% onemat) + (w1 %*% a0)
  if (f=="sigmoid"){a1 = sigmoid(Z)}
  if (f=="relu"){a1 = relu(Z)}
  if (f=="tanh"){a1 = tanh(Z)}
  Z2 = (b2 %*% onemat) + (w2 %*% a1)
  if (g=="identity") {a=identity(Z2)}
  if (g=="sigmoid") {a=sigmoid(Z2)}
  if (g=="stable.softmax") {a=stable.softmax(Z2)}
  return(a)
}

num.gradient=function(cost,x,y,b1,w1,b2,w2,eps=1e-8,f,g){
  dwi=numeric(length(w1))
  dwi2=numeric(length(w2))
  db1=numeric(length(b1))
  db2=numeric(length(b2))
  for (i in 1:length(b1)){
    bp=bm=b1
    bp[i]=bp[i]+eps
    bm[i]=bm[i]-eps
    if (cost=="cost.cross.entropy"){db1[i]=( cost.cross.entropy(x,y,bp,w1,b2,w2,f,g)-cost.cross.entropy(x,y,bm,w1,b2,w2,f,g))/(2*eps)}
    if (cost=="cost.negll"){db1[i]=( cost.negll(x,y,bp,w1,b2,w2,f,g)-cost.negll(x,y,bm,w1,b2,w2,f,g))/(2*eps)}
    if (cost=="cost.squared.error"){db1[i]=( cost.squared.error(x,y,bp,w1,b2,w2,f,g)-cost.squared.error(x,y,bm,w1,b2,w2,f,g))/(2*eps)}
  }
  for (i in 1:length(w1)){
    wp=wm=w1
    wp[i]=wp[i]+eps
    wm[i]=wm[i]-eps
    if (cost=="cost.cross.entropy"){dwi[i]=(cost.cross.entropy(x,y,b1,wp,b2,w2,f,g)-cost.cross.entropy(x,y,b1,wm,b2,w2,f,g))/(2*eps)}
    if (cost=="cost.negll"){dwi[i]=(cost.negll(x,y,b1,wp,b2,w2,f,g)-cost.negll(x,y,b1,wm,b2,w2,f,g))/(2*eps)}
    if (cost=="cost.squared.error"){dwi[i]=(cost.squared.error(x,y,b1,wp,b2,w2,f,g)-cost.squared.error(x,y,b1,wm,b2,w2,f,g))/(2*eps)}
  } 
  for (i in 1:length(b2)){
    bp2=bm2=b2
    bp2[i]=bp2[i]+eps
    bm2[i]=bm2[i]-eps
    if (cost=="cost.cross.entropy"){db2[i]=(cost.cross.entropy(x,y,b1,w1,bp2,w2,f,g)-cost.cross.entropy(x,y,b1,w1,bm2,w2,f,g))/(2*eps)}
    if (cost=="cost.negll"){db2[i]=(cost.negll(x,y,b1,w1,bp2,w2,f,g)-cost.negll(x,y,b1,w1,bm2,w2,f,g))/(2*eps)}
    if (cost=="cost.squared.error"){db2[i]=(cost.squared.error(x,y,b1,w1,bp2,w2,f,g)-cost.squared.error(x,y,b1,w1,bm2,w2,f,g))/(2*eps)}
  }
  for (i in 1:length(w2)){
    wp2=wm2=w2
    wp2[i]=wp2[i]+eps
    wm2[i]=wm2[i]-eps
    if (cost=="cost.cross.entropy"){dwi2[i]=(cost.cross.entropy(x,y,b1,w1,b2,wp2,f,g)-cost.cross.entropy(x,y,b1,w1,b2,wm2,f,g))/(2*eps)}
    if (cost=="cost.negll"){dwi2[i]=(cost.negll(x,y,b1,w1,b2,wp2,f,g)-cost.negll(x,y,b1,w1,b2,wm2,f,g))/(2*eps)}
    if (cost=="cost.squared.error"){dwi2[i]=(cost.squared.error(x,y,b1,w1,b2,wp2,f,g)-cost.squared.error(x,y,b1,w1,b2,wm2,f,g))/(2*eps)}
  } 
  dw1 = matrix(dwi,ncol=ncol(w1),nrow=nrow(w1))
  dw2 = matrix(dwi2,ncol=ncol(w2),nrow=nrow(w2))
  return(list(db1=db1,dw1=dw1,db2=db2,dw2=dw2))
}




fit=function(f,g,x,b1,w1,b2,w2){fwd.prop(f,g,x,b1,w1,b2,w2)}
predict=function(f,g,x,b1,w1,b2,w2){
  s=fit(f,g,x,b1,w1,b2,w2)
  apply(s,2,which.max)
}
accuracy=function(f,g,x,b1,w1,b2,w2,Y){
  pred=predict(f,g,x,b1,w1,b2,w2)
  sum(pred == Y)/length(Y)
}
perf.history=function(xin,y,history){
  perf=function(el){accuracy(xin,y,el$b,el$w)}
  sapply(history,perf)
}
plot.perf.history=function(xin,y,history){
  perf=function(el){accuracy(xin,y,el$b,el$w)}
  plot(sapply(history,perf),main="Performance History",ylab="Performance Accuracy")
}
