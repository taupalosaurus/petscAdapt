#include "petscvec.h"
#include "taosolver.h"
#include "private/taolinesearch_impl.h"
#include "morethuente.h"

static PetscErrorCode Tao_mcstep(TaoLineSearch ls,
				 double *stx, double *fx, double *dx,
				 double *sty, double *fy, double *dy,
				 double *stp, double *fp, double *dp);

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchDestroy_MT"
static PetscErrorCode TaoLineSearchDestroy_MT(TaoLineSearch ls)
{
  PetscErrorCode ierr;
  TAOLINESEARCH_MT_CTX *mt;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
  mt = (TAOLINESEARCH_MT_CTX*)(ls->data);
  if (mt->x) {
    ierr = VecDestroy(mt->x); CHKERRQ(ierr); 
  }
  if (mt->work) {
    ierr = VecDestroy(mt->work); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchSetFromOptions_MT"
static PetscErrorCode TaoLineSearchSetFromOptions_MT(TaoLineSearch ls)
{
  PetscFunctionBegin;
  
  PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchView_MT"
static PetscErrorCode TaoLineSearchView_MT(TaoLineSearch ls, PetscViewer pv)
{
    PetscErrorCode ierr;
    PetscTruth isascii;
    PetscFunctionBegin;
    ierr = PetscTypeCompare((PetscObject)pv, PETSC_VIEWER_ASCII, &isascii); CHKERRQ(ierr);
    if (isascii) {
	ierr = PetscViewerASCIIPrintf(pv,"  Line Search: MoreThuente.\n"); CHKERRQ(ierr);
    } else {
	SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for MoreThuente TaoLineSearch",((PetscObject)pv)->type_name);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchApply_MT"
/* @ TaoApply_LineSearch - This routine takes step length of 1.0.

   Input Parameters:
+  tao - TaoSolver context
.  X - current iterate (on output X contains new iterate, X + step*S)
.  f - objective function evaluated at X
.  G - gradient evaluated at X
-  D - search direction


   Info is set to 0.

@ */

static PetscErrorCode TaoLineSearchApply_MT(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, Vec s) 
{
    PetscErrorCode ierr;
    TAOLINESEARCH_MT_CTX *mt;
#if defined(PETSC_USE_SCALAR)
    PetscReal cdginit;
#endif
    
    PetscReal    xtrapf = 4.0;
    PetscReal   finit, width, width1, dginit, fm, fxm, fym, dgm, dgxm, dgym;
    PetscReal    dgx, dgy, dg, fx, fy, stx, sty, dgtest, ftest1=0.0;
    PetscInt  i, stage1;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    PetscValidHeaderSpecific(x,VEC_COOKIE,2);
    PetscValidScalarPointer(f,3);
    PetscValidHeaderSpecific(g,VEC_COOKIE,4);
    PetscValidHeaderSpecific(s,VEC_COOKIE,5);

    /* comm,type,size checks are done in interface TaoLineSearchApply */
    mt = (TAOLINESEARCH_MT_CTX*)(ls->data);

    /* Check work vector */
    if (!mt->work) {
      ierr = VecDuplicate(x,&mt->work); CHKERRQ(ierr);
      mt->x = x;
      ierr = PetscObjectReference((PetscObject)x); CHKERRQ(ierr);
    }
    /* If x has changed, then recreate work */
    else if (x != mt->x) { 
      ierr = VecDestroy(mt->x); CHKERRQ(ierr);
      ierr = VecDestroy(mt->work); CHKERRQ(ierr);
      ierr = VecDuplicate(x,&mt->work); CHKERRQ(ierr);
      mt->x = x;
      ierr = PetscObjectReference((PetscObject)x); CHKERRQ(ierr);
    }
#if defined(PETSC_USE_COMPLEX)
    ierr = VecDot(g,s,&cdginit); CHKERRQ(ierr); dginit = PetscReal(cdginit);
#else
    ierr = VecDot(g,s,&dginit);
#endif
    
    if (PetscIsInfOrNanReal(dginit)) {
      ierr = PetscInfo1(ls,"Initial Line Search step * g is Inf or Nan (%g)\n",dginit); CHKERRQ(ierr);
      ls->reason=TAOLINESEARCH_FAILED_INFORNAN;
      PetscFunctionReturn(0);
    }
    if (dginit >= 0.0) {
      ierr = PetscInfo1(ls,"Initial Line Search step * g is not descent direction (%g)\n",dginit); CHKERRQ(ierr);
      ls->reason = TAOLINESEARCH_FAILED_ASCENT;
      PetscFunctionReturn(0);
    }


    /* Initialization */
    mt->bracket = 0;
    stage1 = 1;
    finit = *f;
    dgtest = ls->ftol * dginit;
    width = ls->stepmax - ls->stepmin;
    width1 = width * 2.0;
    ierr = VecCopy(x,mt->work);CHKERRQ(ierr);
    /* Variable dictionary:  
     stx, fx, dgx - the step, function, and derivative at the best step
     sty, fy, dgy - the step, function, and derivative at the other endpoint 
                   of the interval of uncertainty
     step, f, dg - the step, function, and derivative at the current step */

    stx = 0.0;
    fx  = finit;
    dgx = dginit;
    sty = 0.0;
    fy  = finit;
    dgy = dginit;
 
    ls->nfev = 0;
    ls->step=1.0;
    for (i=0; i< ls->maxfev; i++) {
    /* Set min and max steps to correspond to the interval of uncertainty */
      if (mt->bracket) {
	ls->stepmin = PetscMin(stx,sty); 
	ls->stepmax = PetscMax(stx,sty); 
      } 
      else {
	ls->stepmin = stx;
	ls->stepmax = ls->step + xtrapf * (ls->step - stx);
      }

      /* Force the step to be within the bounds */
      ls->step = PetscMax(ls->step,ls->stepmin);
      ls->step = PetscMin(ls->step,ls->stepmax);
    
      /* If an unusual termination is to occur, then let step be the lowest
	 point obtained thus far */
      if (((mt->bracket) && (ls->step <= ls->stepmin || ls->step >= ls->stepmax)) ||
        ((mt->bracket) && (ls->stepmax - ls->stepmin <= ls->rtol * ls->stepmax)) ||
        (ls->nfev >= ls->maxfev - 1) || (mt->infoc == 0)) {
	ls->step = stx;
      }

#if defined(PETSC_USE_COMPLEX)
      cstep = ls->step;
      ierr = VecCopy(mt->work,x); CHKERRQ(ierr);
      ierr = VecAXPY(x,cstep,s); CHKERRQ(ierr);
#else
      ierr = VecCopy(mt->work,x); CHKERRQ(ierr);
      ierr = VecAXPY(x,ls->step,s); CHKERRQ(ierr);
//    info = X->Waxpby(*step,S,1.0,W);CHKERRQ(info); 	/* X = W + step*S */
#endif
      ierr = TaoLineSearchComputeObjectiveAndGradient(ls,x,f,g); CHKERRQ(ierr);
      if (0 == i) {
	ls->f_fullstep=*f;
      }

      ls->nfev++;
#if defined(PETSC_USE_COMPLEX)
      ierr = VecDot(g,s,&cdg); CHKERRQ(ierr); dg = PetscReal(cdg);
//    info = G->Dot(S,&cdg);CHKERRQ(info); dg = TaoReal(cdg);
#else
      ierr = VecDot(g,s,&dg); CHKERRQ(ierr);
//    info = G->Dot(S,&dg);CHKERRQ(info);	        /* dg = G^T S */
#endif

      if (PetscIsInfOrNanReal(*f) || PetscIsInfOrNanReal(dg)) {
	// User provided compute function generated Not-a-Number, assume 
	// domain violation and set function value and directional
	// derivative to infinity.
	*f = TAO_INFINITY;
	dg = TAO_INFINITY;
      }

      ftest1 = finit + ls->step * dgtest;

      /* Convergence testing */
      /* TODO  make parameter for 1.0e-10 */
      if (((*f - ftest1 <= 1.0e-10 * PetscAbsReal(finit)) && 
	   (PetscAbsReal(dg) + ls->gtol*dginit <= 0.0))) {
	ierr = PetscInfo(ls, "TaoApply_LineSearch:Line search success: Sufficient decrease and directional deriv conditions hold\n"); CHKERRQ(ierr);
	ls->reason = TAOLINESEARCH_SUCCESS;
	break;
      }

      /* Checks for bad cases */
      if (((mt->bracket) && (ls->step <= ls->stepmin||ls->step >= ls->stepmax)) || (!mt->infoc)) {
	ierr = PetscInfo(ls,"Rounding errors may prevent further progress.  May not be a step satisfying\n"); CHKERRQ(ierr);
	ierr = PetscInfo(ls,"sufficient decrease and curvature conditions. Tolerances may be too small.\n"); CHKERRQ(ierr);
	ls->reason = TAOLINESEARCH_FAILED_OTHER;
	break;
      }
      if ((ls->step == ls->stepmax) && (*f <= ftest1) && (dg <= dgtest)) {
	ierr = PetscInfo1(ls,"Step is at the upper bound, stepmax (%g)\n",ls->stepmax); CHKERRQ(ierr);
	ls->reason = TAOLINESEARCH_FAILED_UPPERBOUND;
	break;
      }
      if ((ls->step == ls->stepmin) && (*f >= ftest1) && (dg >= dgtest)) {
	ierr = PetscInfo1(ls,"Step is at the lower bound, stepmin (%g)\n",ls->stepmin); CHKERRQ(ierr);
	ls->reason = TAOLINESEARCH_FAILED_LOWERBOUND;
	break;
      }
      if (ls->nfev >= ls->maxfev) {
	ierr = PetscInfo2(ls,"Number of line search function evals (%d) > maximum (%d)\n",ls->nfev,ls->maxfev); CHKERRQ(ierr);
	ls->reason = TAOLINESEARCH_FAILED_MAXFCN;
	break;
      }
      if ((mt->bracket) && (ls->stepmax - ls->stepmin <= ls->rtol*ls->stepmax)){
	ierr = PetscInfo1(ls,"Relative width of interval of uncertainty is at most rtol (%g)\n",ls->rtol); CHKERRQ(ierr);
	ls->reason = TAOLINESEARCH_FAILED_RTOL;
	break;
      }

      /* In the first stage, we seek a step for which the modified function
	 has a nonpositive value and nonnegative derivative */
      if ((stage1) && (*f <= ftest1) && (dg >= dginit * PetscMin(ls->ftol, ls->gtol))) {
	stage1 = 0;
      }

      /* A modified function is used to predict the step only if we
	 have not obtained a step for which the modified function has a 
	 nonpositive function value and nonnegative derivative, and if a
	 lower function value has been obtained but the decrease is not
	 sufficient */

      if ((stage1) && (*f <= fx) && (*f > ftest1)) {
	fm   = *f - ls->step * dgtest;	/* Define modified function */
	fxm  = fx - stx * dgtest;	        /* and derivatives */
	fym  = fy - sty * dgtest;
	dgm  = dg - dgtest;
	dgxm = dgx - dgtest;
	dgym = dgy - dgtest;
      
	/* Update the interval of uncertainty and compute the new step */
	ierr = Tao_mcstep(ls,&stx,&fxm,&dgxm,&sty,&fym,&dgym,&ls->step,&fm,&dgm);CHKERRQ(ierr);
      
	fx  = fxm + stx * dgtest;	/* Reset the function and */
	fy  = fym + sty * dgtest;	/* gradient values */
	dgx = dgxm + dgtest; 
	dgy = dgym + dgtest; 
      } 
      else {
	/* Update the interval of uncertainty and compute the new step */
	ierr = Tao_mcstep(ls,&stx,&fx,&dgx,&sty,&fy,&dgy,&ls->step,f,&dg);CHKERRQ(ierr);
      }
    
      /* Force a sufficient decrease in the interval of uncertainty */
      if (mt->bracket) {
	if (PetscAbsReal(sty - stx) >= 0.66 * width1) ls->step = stx + 0.5*(sty - stx);
	width1 = width;
	width = PetscAbsReal(sty - stx);
      }
    }
  
    /* Finish computations */
    ierr = PetscInfo2(ls,"%d function evals in line search, step = %10.4f\n",ls->nfev,ls->step); CHKERRQ(ierr);
    
      
    PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TaoLineSearchCreate_MT"
PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchCreate_MT(TaoLineSearch ls)
{
    PetscErrorCode ierr;
    TAOLINESEARCH_MT_CTX *ctx;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(ls,TAOLINESEARCH_COOKIE,1);
    ierr = PetscNewLog(ls,TAOLINESEARCH_MT_CTX,&ctx); CHKERRQ(ierr);
    ctx->bracket=0;
    ctx->infoc=1;
    ls->data = (void*)ctx;
    ls->ops->setup=0; //TaoLineSearchSetup_MT;
    ls->ops->apply=TaoLineSearchApply_MT;
    ls->ops->view =TaoLineSearchView_MT;
    ls->ops->destroy=TaoLineSearchDestroy_MT;
    ls->ops->setfromoptions=TaoLineSearchSetFromOptions_MT;
    PetscFunctionReturn(0);
}
EXTERN_C_END



/*
     The subroutine mcstep is taken from the work of Jorge Nocedal.
     this is a variant of More' and Thuente's routine.

     subroutine mcstep

     the purpose of mcstep is to compute a safeguarded step for
     a linesearch and to update an interval of uncertainty for
     a minimizer of the function.

     the parameter stx contains the step with the least function
     value. the parameter stp contains the current step. it is
     assumed that the derivative at stx is negative in the
     direction of the step. if bracket is set true then a
     minimizer has been bracketed in an interval of uncertainty
     with endpoints stx and sty.

     the subroutine statement is

     subroutine mcstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,bracket,
                       stpmin,stpmax,info)

     where

       stx, fx, and dx are variables which specify the step,
         the function, and the derivative at the best step obtained
         so far. The derivative must be negative in the direction
         of the step, that is, dx and stp-stx must have opposite
         signs. On output these parameters are updated appropriately.

       sty, fy, and dy are variables which specify the step,
         the function, and the derivative at the other endpoint of
         the interval of uncertainty. On output these parameters are
         updated appropriately.

       stp, fp, and dp are variables which specify the step,
         the function, and the derivative at the current step.
         If bracket is set true then on input stp must be
         between stx and sty. On output stp is set to the new step.

       bracket is a logical variable which specifies if a minimizer
         has been bracketed.  If the minimizer has not been bracketed
         then on input bracket must be set false.  If the minimizer
         is bracketed then on output bracket is set true.

       stpmin and stpmax are input variables which specify lower
         and upper bounds for the step.

       info is an integer output variable set as follows:
         if info = 1,2,3,4,5, then the step has been computed
         according to one of the five cases below. otherwise
         info = 0, and this indicates improper input parameters.

     subprograms called

       fortran-supplied ... abs,max,min,sqrt

     argonne national laboratory. minpack project. june 1983
     jorge j. more', david j. thuente

*/

#undef __FUNCT__  
#define __FUNCT__ "TaoStep_LineSearch"
static PetscErrorCode Tao_mcstep(TaoLineSearch ls,
                              double *stx, double *fx, double *dx,
		              double *sty, double *fy, double *dy,
			      double *stp, double *fp, double *dp)
{
  TAOLINESEARCH_MT_CTX *mtP = (TAOLINESEARCH_MT_CTX *) ls->data;
  PetscReal gamma1, p, q, r, s, sgnd, stpc, stpf, stpq, theta;
  PetscInt bound;

  PetscFunctionBegin;

  // Check the input parameters for errors
  mtP->infoc = 0;
  if (mtP->bracket && (*stp <= PetscMin(*stx,*sty) || (*stp >= PetscMax(*stx,*sty)))) SETERRQ(1,"bad stp in bracket");
  if (*dx * (*stp-*stx) >= 0.0) SETERRQ(1,"dx * (stp-stx) >= 0.0");
  if (ls->stepmax < ls->stepmin) SETERRQ(1,"stepmax > stepmin");

  // Determine if the derivatives have opposite sign */
  sgnd = *dp * (*dx / PetscAbsReal(*dx));

  if (*fp > *fx) {
    // Case 1: a higher function value.
    // The minimum is bracketed. If the cubic step is closer
    // to stx than the quadratic step, the cubic step is taken,
    // else the average of the cubic and quadratic steps is taken.

    mtP->infoc = 1;
    bound = 1;
    theta = 3 * (*fx - *fp) / (*stp - *stx) + *dx + *dp;
    s = PetscMax(PetscAbsReal(theta),PetscAbsReal(*dx));
    s = PetscMax(s,PetscAbsReal(*dp));
    gamma1 = s*sqrt(pow(theta/s,2.0) - (*dx/s)*(*dp/s));
    if (*stp < *stx) gamma1 = -gamma1;
    /* Can p be 0?  Check */
    p = (gamma1 - *dx) + theta;
    q = ((gamma1 - *dx) + gamma1) + *dp;
    r = p/q;
    stpc = *stx + r*(*stp - *stx);
    stpq = *stx + ((*dx/((*fx-*fp)/(*stp-*stx)+*dx))*0.5) * (*stp - *stx);

    if (PetscAbsReal(stpc-*stx) < PetscAbsReal(stpq-*stx)) {
      stpf = stpc;
    } 
    else {
      stpf = stpc + 0.5*(stpq - stpc);
    }
    mtP->bracket = 1;
  }
  else if (sgnd < 0.0) {
    // Case 2: A lower function value and derivatives of
    // opposite sign. The minimum is bracketed. If the cubic
    // step is closer to stx than the quadratic (secant) step,
    // the cubic step is taken, else the quadratic step is taken.

    mtP->infoc = 2;
    bound = 0;
    theta = 3*(*fx - *fp)/(*stp - *stx) + *dx + *dp;
    s = PetscMax(PetscAbsReal(theta),PetscAbsReal(*dx));
    s = PetscMax(s,PetscAbsReal(*dp));
    gamma1 = s*sqrt(pow(theta/s,2.0) - (*dx/s)*(*dp/s));
    if (*stp > *stx) gamma1 = -gamma1;
    p = (gamma1 - *dp) + theta;
    q = ((gamma1 - *dp) + gamma1) + *dx;
    r = p/q;
    stpc = *stp + r*(*stx - *stp);
    stpq = *stp + (*dp/(*dp-*dx))*(*stx - *stp);

    if (PetscAbsReal(stpc-*stp) > PetscAbsReal(stpq-*stp)) {
      stpf = stpc;
    }
    else {
      stpf = stpq;
    }
    mtP->bracket = 1;
  }
  else if (PetscAbsReal(*dp) < PetscAbsReal(*dx)) {
    // Case 3: A lower function value, derivatives of the
    // same sign, and the magnitude of the derivative decreases.
    // The cubic step is only used if the cubic tends to infinity
    // in the direction of the step or if the minimum of the cubic
    // is beyond stp. Otherwise the cubic step is defined to be
    // either stepmin or stepmax. The quadratic (secant) step is also
    // computed and if the minimum is bracketed then the the step
    // closest to stx is taken, else the step farthest away is taken.

    mtP->infoc = 3;
    bound = 1;
    theta = 3*(*fx - *fp)/(*stp - *stx) + *dx + *dp;
    s = PetscMax(PetscAbsReal(theta),PetscAbsReal(*dx));
    s = PetscMax(s,PetscAbsReal(*dp));

    // The case gamma1 = 0 only arises if the cubic does not tend
    // to infinity in the direction of the step.
    gamma1 = s*sqrt(PetscMax(0.0,pow(theta/s,2.0) - (*dx/s)*(*dp/s)));
    if (*stp > *stx) gamma1 = -gamma1;
    p = (gamma1 - *dp) + theta;
    q = (gamma1 + (*dx - *dp)) + gamma1;
    r = p/q;
    if (r < 0.0 && gamma1 != 0.0) stpc = *stp + r*(*stx - *stp);
    else if (*stp > *stx)        stpc = ls->stepmax;
    else                         stpc = ls->stepmin;
    stpq = *stp + (*dp/(*dp-*dx)) * (*stx - *stp);

    if (mtP->bracket) {
      if (PetscAbsReal(*stp-stpc) < PetscAbsReal(*stp-stpq)) {
	stpf = stpc;
      } 
      else {
	stpf = stpq;
      }
    }
    else {
      if (PetscAbsReal(*stp-stpc) > PetscAbsReal(*stp-stpq)) {
	stpf = stpc;
      }
      else {
	stpf = stpq;
      }
    }
  }
  else {
    // Case 4: A lower function value, derivatives of the
    // same sign, and the magnitude of the derivative does
    // not decrease. If the minimum is not bracketed, the step
    // is either stpmin or stpmax, else the cubic step is taken.

    mtP->infoc = 4;
    bound = 0;
    if (mtP->bracket) {
      theta = 3*(*fp - *fy)/(*sty - *stp) + *dy + *dp;
      s = PetscMax(PetscAbsReal(theta),PetscAbsReal(*dy));
      s = PetscMax(s,PetscAbsReal(*dp));
      gamma1 = s*sqrt(pow(theta/s,2.0) - (*dy/s)*(*dp/s));
      if (*stp > *sty) gamma1 = -gamma1;
      p = (gamma1 - *dp) + theta;
      q = ((gamma1 - *dp) + gamma1) + *dy;
      r = p/q;
      stpc = *stp + r*(*sty - *stp);
      stpq = *stp + (*dp/(*dp-*dx)) * (*stx - *stp);

      stpf = stpc;
    } 
    else if (*stp > *stx) {
      stpf = ls->stepmax;
    } 
    else {
      stpf = ls->stepmin;
    }
  }
  
  // Update the interval of uncertainty.  This update does not
  // depend on the new step or the case analysis above.

  if (*fp > *fx) {
    *sty = *stp;
    *fy = *fp;
    *dy = *dp;
  } 
  else {
    if (sgnd < 0.0) {
      *sty = *stx;
      *fy = *fx;
      *dy = *dx;
    }
    *stx = *stp;
    *fx = *fp;
    *dx = *dp;
  }
  
  // Compute the new step and safeguard it.
  stpf = PetscMin(ls->stepmax,stpf);
  stpf = PetscMax(ls->stepmin,stpf);
  *stp = stpf;
  if (mtP->bracket && bound) {
    if (*sty > *stx) {
      *stp = PetscMin(*stx+0.66*(*sty-*stx),*stp);
    }
    else {
      *stp = PetscMax(*stx+0.66*(*sty-*stx),*stp);
    }
  }
  PetscFunctionReturn(0);
}
