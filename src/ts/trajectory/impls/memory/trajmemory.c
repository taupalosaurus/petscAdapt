
#include <petsc/private/tsimpl.h>        /*I "petscts.h"  I*/
#include <petscsys.h>

extern int wrap_revolve(int* check,int* capo,int* fine,int *snaps_in,int* info);

typedef struct _StackElement {
  PetscInt  stepnum;
  Vec       X;
  Vec       *Y;
  PetscReal time;
  PetscReal timeprev;
} *StackElement; 

typedef struct _Stack {
   PetscBool    usecontroller;
   PetscBool    reverseonestep;
   PetscInt     stepsleft;
   PetscInt     check;
   PetscInt     capo;
   PetscInt     fine;
   PetscInt     info;
   PetscInt     top;         /* The top of the stack */
   PetscInt     maxelements; /* The maximum stack size */
   PetscInt     numY;
   MPI_Comm     comm;
   StackElement *stack;      /* The storage */
 } Stack;

static PetscErrorCode StackCreate(MPI_Comm,Stack *,PetscInt,PetscInt);
static PetscErrorCode StackDestroy(Stack*);
static PetscErrorCode StackPush(Stack*,StackElement);
static PetscErrorCode StackPop(Stack*,StackElement*);
static PetscErrorCode StackTop(Stack*,StackElement*);

#undef __FUNCT__
#define __FUNCT__ "StackCreate"
static PetscErrorCode StackCreate(MPI_Comm comm,Stack *s,PetscInt size,PetscInt ny)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  s->top         = -1;
  s->maxelements = size;
  s->comm        = comm;
  s->numY        = ny;

  ierr = PetscMalloc1(s->maxelements*sizeof(StackElement),&s->stack);CHKERRQ(ierr);
  ierr = PetscMemzero(s->stack,s->maxelements*sizeof(StackElement));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackDestroy"
static PetscErrorCode StackDestroy(Stack *s)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->top>-1) {
    for (i=0;i<=s->top;i++) {
      ierr = VecDestroy(&s->stack[i]->X);CHKERRQ(ierr);
      ierr = VecDestroyVecs(s->numY,&s->stack[i]->Y);CHKERRQ(ierr);
      ierr = PetscFree(s->stack[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(s->stack);CHKERRQ(ierr);
  ierr = PetscFree(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackPush"
static PetscErrorCode StackPush(Stack *s,StackElement e)
{
  PetscFunctionBegin;
  if (s->top+1 >= s->maxelements) SETERRQ1(s->comm,PETSC_ERR_MEMC,"Maximum stack size (%D) exceeded",s->maxelements);
  s->stack[++s->top] = e;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackPop"
static PetscErrorCode StackPop(Stack *s,StackElement *e)
{
  PetscFunctionBegin;
  if (s->top == -1) SETERRQ(s->comm,PETSC_ERR_MEMC,"Emptry stack");
  *e = s->stack[s->top--];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StackTop"
static PetscErrorCode StackTop(Stack *s,StackElement *e)
{
  PetscFunctionBegin;
  *e = s->stack[s->top];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectorySet_Memory"
PetscErrorCode TSTrajectorySet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal time,Vec X)
{
  PetscInt       ns,i;
  Vec            *Y;
  PetscReal      timeprev;
  StackElement   e;
  Stack          *s = (Stack*)tj->data;
  PetscInt       whattodo,oldcapo;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (stepnum<s->top) SETERRQ(s->comm,PETSC_ERR_MEMC,"Illegal modification of a non-top stack element");
  
  if (s->usecontroller) {
    printf("left %d\n",s->stepsleft);
    if (s->reverseonestep) PetscFunctionReturn(0);
    if (s->stepsleft==0) { /* let the controller determine what to do next */
      s->capo = stepnum;
      oldcapo = s->capo;
      printf("check=%d,capo=%d,fine=%d,snaps_in=%d\t",s->check,s->capo,s->fine,s->maxelements);
      whattodo = wrap_revolve(&s->check,&s->capo,&s->fine,&s->maxelements,&s->info);
switch(whattodo) {
      case 1:
        printf("Advance from %d to %d.\n", oldcapo, s->capo);
        break;
      case 2:
        printf("Store in checkpoint number %d\n",s->check);
        break;
      case 3:
        printf("First turn: Initialize adjoints and reverse first step.\n");
        break;
      case 4:
        printf("Forward and reverse one step.\n");
        break;
      case 5:
        printf("Restore in checkpoint number %d\n",s->check);
        break;
      case -1:
        printf("Error!");
        break;
}
      if (whattodo==-1) SETERRQ(s->comm,PETSC_ERR_MEMC,"Error in the controller");
      if (whattodo==1) {
        s->stepsleft = s->capo-oldcapo-1;
        PetscFunctionReturn(0); /* do not need to checkpoint */ 
      }
      if (whattodo==3 || whattodo==4) {
        s->reverseonestep = PETSC_TRUE;
        PetscFunctionReturn(0);
      }
      if (whattodo==5) {
        printf("check=%d,capo=%d,fine=%d,snaps_in=%d\t",s->check,s->capo,s->fine,s->maxelements);
        oldcapo = s->capo;
        whattodo = wrap_revolve(&s->check,&s->capo,&s->fine,&s->maxelements,&s->info);
        if (whattodo!=1) SETERRQ(s->comm,PETSC_ERR_MEMC,"Something weird happened");
        s->stepsleft = s->capo-oldcapo;
switch(whattodo) {
      case 1:
        printf("5Advance from %d to %d.\n", oldcapo, s->capo);
        break;
      case 2:
        printf("5Store in checkpoint number %d\n",s->check);
        break;
      case 3:
        printf("5First turn: Initialize adjoints and reverse first step.\n");
        break;
      case 4:
        printf("5Forward and reverse one step.\n");
        break;
      case 5:
        printf("5Restore in checkpoint number %d\n",s->check);
        break;
      case -1:
        printf("Error!");
        break;
}
        PetscFunctionReturn(0);
      }
      if (whattodo==2) {
        printf("check=%d,capo=%d,fine=%d,snaps_in=%d\t",s->check,s->capo,s->fine,s->maxelements);
        oldcapo = s->capo;
        whattodo = wrap_revolve(&s->check,&s->capo,&s->fine,&s->maxelements,&s->info);
        if (whattodo!=1) SETERRQ(s->comm,PETSC_ERR_MEMC,"Something weird happened");
        if (stepnum==0) s->stepsleft = s->capo-oldcapo-1;
        else s->stepsleft = s->capo-oldcapo-1;
switch(whattodo) {
      case 1:
        printf("2Advance from %d to %d.\n", oldcapo, s->capo);
        break;
      case 2:
        printf("2Store in checkpoint number %d\n",s->check);
        break;
      case 3:
        printf("2First turn: Initialize adjoints and reverse first step.\n");
        break;
      case 4:
        printf("2Forward and reverse one step.\n");
        break;
      case 5:
        printf("2Restore in checkpoint number %d\n",s->check);
        break;
      case -1:
        printf("Error!");
        break;
}
      }
    } else { /* advance s->stepsleft time steps without checkpointing */
      s->stepsleft--;
      PetscFunctionReturn(0);
    }
  }
 
  /* checkpoint to memmory */
  if (stepnum==s->top) { /* overwrite the top checkpoint */
    ierr = StackTop(s,&e); 
    ierr = VecCopy(X,e->X);CHKERRQ(ierr);
    ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);
    for (i=0;i<ns;i++) {
      ierr = VecCopy(Y[i],e->Y[i]);CHKERRQ(ierr);
    }
    e->stepnum  = stepnum;
    e->time     = time;
    ierr        = TSGetPrevTime(ts,&timeprev);CHKERRQ(ierr);
    e->timeprev = timeprev;
  } else {
    ierr = PetscCalloc1(1,&e);
    ierr = VecDuplicate(X,&e->X);CHKERRQ(ierr);
    ierr = VecCopy(X,e->X);CHKERRQ(ierr);
    ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(Y[0],ns,&e->Y);CHKERRQ(ierr);
    for (i=0;i<ns;i++) {
      ierr = VecCopy(Y[i],e->Y[i]);CHKERRQ(ierr);
    }
    e->stepnum  = stepnum;
    e->time     = time;
    if (stepnum == 0) {
      e->timeprev = e->time - ts->time_step; /* for consistency */
    } else {
      ierr        = TSGetPrevTime(ts,&timeprev);CHKERRQ(ierr);
      e->timeprev = timeprev;
    }
    ierr        = StackPush(s,e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryGet_Memory"
PetscErrorCode TSTrajectoryGet_Memory(TSTrajectory tj,TS ts,PetscInt stepnum,PetscReal *t)
{
  Vec            *Y;
  PetscInt       nr,i;
  StackElement   e;
  Stack          *s = (Stack*)tj->data;
  PetscReal      stepsize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (s->usecontroller && s->reverseonestep) {
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr); /* go backward */
    s->reverseonestep = PETSC_FALSE;
    printf("reverse step %d\n",stepnum);
    PetscFunctionReturn(0);
  }

  /* restore a checkpoint */
  ierr = StackTop(s,&e);CHKERRQ(ierr);
  ierr = VecCopy(e->X,ts->vec_sol);CHKERRQ(ierr);
  ierr = TSGetStages(ts,&nr,&Y);CHKERRQ(ierr);
  for (i=0;i<nr ;i++) {
    ierr = VecCopy(e->Y[i],Y[i]);CHKERRQ(ierr);
  }
  *t = e->time;

  printf("read stepnum=%d to get %d\n",e->stepnum,stepnum);
  if (e->stepnum < stepnum) { /* need recomputation */
    PetscInt whattodo;
    s->capo = stepnum;
    printf("check=%d,capo=%d,fine=%d,snaps_in=%d\t",s->check,s->capo,s->fine,s->maxelements);
    whattodo = wrap_revolve(&s->check,&s->capo,&s->fine,&s->maxelements,&s->info);
switch(whattodo) {
      case 5:
        printf("R:Restore in checkpoint number %d\n",s->check);
        break;
      case -1:
        printf("Error!");
        break;
}
    ierr = TSSetTimeStep(ts,(*t)-e->timeprev);CHKERRQ(ierr);
    /* reset ts context */
    PetscInt steps = ts->steps;
    ts->steps      = e->stepnum; 
    ts->ptime      = e->time;
    ts->ptime_prev = e->timeprev; 
    printf("need to recompute %d steps\n",stepnum-e->stepnum);
    for (i=e->stepnum;i<stepnum;i++) { /* assume fixed step size */
      ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
      ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
      ierr = TSStep(ts);CHKERRQ(ierr);
      if (ts->event) {
        ierr = TSEventMonitor(ts);CHKERRQ(ierr);
      }
      if (!ts->steprollback) {
        ierr = TSPostStep(ts);CHKERRQ(ierr);
      }
    }
    if (!s->reverseonestep) printf("error\n");
    ts->steps = steps;
    ts->total_steps = stepnum;
    ierr = TSGetTimeStep(ts,&stepsize);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,-stepsize);CHKERRQ(ierr); /* go backward */
    if (stepnum-e->stepnum==1) {
      ierr = StackPop(s,&e);CHKERRQ(ierr);
      ierr = VecDestroy(&e->X);CHKERRQ(ierr);
      ierr = VecDestroyVecs(s->numY,&e->Y);CHKERRQ(ierr);
      ierr = PetscFree(e);CHKERRQ(ierr);
    }
    s->reverseonestep = PETSC_FALSE;
    printf("reverse step %d\n",stepnum);
  } else if (e->stepnum == stepnum) {
    ierr = TSSetTimeStep(ts,-(*t)+e->timeprev);CHKERRQ(ierr); /* go backward */
    ierr = StackPop(s,&e);CHKERRQ(ierr);
    ierr = VecDestroy(&e->X);CHKERRQ(ierr);
    ierr = VecDestroyVecs(s->numY,&e->Y);CHKERRQ(ierr);
    ierr = PetscFree(e);CHKERRQ(ierr);
  } else {
    SETERRQ2(s->comm,PETSC_ERR_ARG_OUTOFRANGE,"The current step no. is %D, but the step number at top of the stack is %D",stepnum,e->stepnum);    
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryDestroy_Memory"
PETSC_EXTERN PetscErrorCode TSTrajectoryDestroy_Memory(TSTrajectory tj)
{
  Stack          *s = (Stack*)tj->data;
  PetscErrorCode ierr; 

  PetscFunctionBegin;
  ierr = StackDestroy(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSTRAJECTORYMEMORY - Stores each solution of the ODE/ADE in memory

  Level: intermediate

.seealso:  TSTrajectoryCreate(), TS, TSTrajectorySetType()

M*/
#undef __FUNCT__
#define __FUNCT__ "TSTrajectoryCreate_Memory"
PETSC_EXTERN PetscErrorCode TSTrajectoryCreate_Memory(TSTrajectory tj,TS ts)
{
  PetscInt       nr,maxsteps;
  Stack          *s;
  PetscErrorCode ierr;
 
  PetscFunctionBegin;
  tj->ops->set     = TSTrajectorySet_Memory;
  tj->ops->get     = TSTrajectoryGet_Memory;
  tj->ops->destroy = TSTrajectoryDestroy_Memory;
 
  ierr = PetscCalloc1(1,&s);
  s->maxelements = 3; /* will be provided by users */
  ierr = TSGetStages(ts,&nr,PETSC_IGNORE);CHKERRQ(ierr);
   
  maxsteps = PetscMin(ts->max_steps,(PetscInt)(ceil(ts->max_time/ts->time_step)));
  if (s->maxelements-1<maxsteps) { /* Need to use a controller */
    s->usecontroller  = PETSC_TRUE;
    s->reverseonestep = PETSC_FALSE;
    s->check          = -1;
    s->capo           = 0;
    s->fine           = maxsteps; 
    s->info           = 2;
    ierr = StackCreate(PetscObjectComm((PetscObject)ts),s,s->maxelements,nr);CHKERRQ(ierr);
  } else { /* Enough space for checkpointing all time steps */
    s->usecontroller = PETSC_FALSE;
    ierr = StackCreate(PetscObjectComm((PetscObject)ts),s,ts->max_steps+1,nr);CHKERRQ(ierr);
  }
  tj->data = s;
  PetscFunctionReturn(0);
}
