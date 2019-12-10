
banana <- function(x){
   100*(x[2]-x[1]^2)^2+(1-x[1])^2
 }

dx1= function(x){(400*x[1]*(x[2]-x[1]^2))-2*(1-x[1])}
dx2 = function(x){200*(x[2]-x[1]^2)}
grad<-function(x) return(c(dx1(x),dx2(x)))

sed <- function (x0, f, g = NULL, info = FALSE,
                           maxiter = 10000000, tol = 1e-7) {
    eps <- 1e-7
    if (! is.numeric(x0))
        stop("Argument 'x0' must be a numeric vector.")
    n <- length(x0)

    # User provided or numerical gradient
    f <- match.fun(f)
    g<- match.fun(grad)

    if (info) cat(0, "\t", x0, "\n")

    x <- x0
    k <- 1
    fixstep <- 0.01
    while (k <= maxiter) {
        f1 <- f(x) #banana function
        g1 <- g(x) #gradient of the banana
        z1 <- sqrt(sum(g1^2)) #norm of the gradient of the banana
        if (z1 == 0) {
            warning(
                paste("Zero gradient at:", x, f1, "-- not applicable.\n"))
            return(list(xmin = NA, fmin = NA, niter = k))
        }
        # else use gradient vector
        g2 <- g1 / z1

        fixstep=0.01; f3 <- f(x - fixstep*g2) #one step ahead to make sure f3 is the lowest before going to the next x value

        # Find a minimum and know when to stop
        if (f3>=f1 && ((abs(f3-f1)<tol)  || (sqrt(sum(x-(x-fixstep*g2))^2))<tol )) {
            f3 <- f1
            
           #if ( z1<tol) {
               # if (info)
               #     cat("Method of steepest descent converged to:", x, "\n")
            #x[x < eps] <- 0
            return(list(xmin = x, fmin = f(x), niter = k))
            #}
           # return(f3)
        }


        x <- x - fixstep*g2
        #therefore, x starts over again, and the new x will be put into the function to ensure banana stays minimum

        k <- k + 1
    }
    if(k > maxiter)
        warning("Maximum number of iterations reached -- not converged.\n")
    return(list(xmin = x, fmin = f(x), niter = k))
}




