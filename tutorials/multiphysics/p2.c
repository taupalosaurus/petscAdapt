
static char help[] = "Model single-physics solver. Modified from ex19.c \n\\n";

/* ------------------------------------------------------------------------

    See ex19.c for discussion of the problem 

  ------------------------------------------------------------------------- */
#include "mp.h"

extern PetscErrorCode FormInitialGuess(DMMG,Vec);
extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscLogEvent  EVENT_FORMFUNCTIONLOCAL2;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;          /* multilevel grid structure */
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,its;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  SNES           snes;
  DM             da2;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscLogEventRegister("FormFunc2", 0,&EVENT_FORMFUNCTIONLOCAL2);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  /* Problem parameters (velocity of lid, prandtl, and grashof numbers) */
  ierr = PetscOptionsGetReal(PETSC_NULL,"-lidvelocity",&user.lidvelocity,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-prandtl",&user.prandtl,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-grashof",&user.grashof,PETSC_NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create user context, set problem data, create vector data structures.
     Also, compute the initial guess.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup Physics 2: 
        - Lap(T) + PR*Div([U*T,V*T]) = 0        
        where U and V are given by the given x.u and x.v
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(comm,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da2);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da2,0,"temperature");CHKERRQ(ierr);

  /* Create the solver object and attach the grid/physics info */
  ierr = DMMGCreate(comm,1,&user,&dmmg);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg,da2);CHKERRQ(ierr);
  ierr = DMMGSetISColoringType(dmmg,IS_COLORING_GLOBAL);CHKERRQ(ierr);

  ierr = DMMGSetInitialGuess(dmmg,FormInitialGuess);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg,FormFunction,0);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da2,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  user.lidvelocity = 1.0/(mx*my);
  user.prandtl     = 1.0;
  user.grashof     = 1.0;

  /* Solve the nonlinear system */
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 
  snes = DMMGGetSNES(dmmg);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Physics 2: Number of Newton iterations = %D\n\n", its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free spaces 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDestroy(da2);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
PetscErrorCode FormInitialGuess(DMMG dmmg,Vec X)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DM             da2 = dmmg->dm;
  Field2         **x2;
  DMDALocalInfo    info2;

  PetscFunctionBegin;
  /* Access the arrays inside  of X */
  ierr = DMDAVecGetArray(da2,X,(void**)&x2);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocal2(&info2,x2,user);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da2,X,(void**)&x2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DMMG           dmmg = (DMMG)ctx;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DM             da2 = dmmg->dm;
  DMDALocalInfo    info2;
  Field2         **x2,**f2;
  Vec            X2;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Get local vectors to hold ghosted parts of X */
  ierr = DMGetLocalVector(da2,&X2);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da2,X,INSERT_VALUES,X2);CHKERRQ(ierr); 
  ierr = DMGlobalToLocalEnd(da2,X,INSERT_VALUES,X2);CHKERRQ(ierr); 

  /* Access the array inside of X1 */
  ierr = DMDAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);

  /* Access the subvectors in F. 
     These are not ghosted so directly access the memory locations in F */
  ierr = DMDAVecGetArray(da2,F,(void**)&f2);CHKERRQ(ierr);

  /* Evaluate local user provided function */    
  ierr = FormFunctionLocal2(&info2,0,x2,f2,(void**)user);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da2,F,(void**)&f2);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da2,&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

