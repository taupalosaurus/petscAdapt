#include <private/linesearchimpl.h>
#include <private/snesimpl.h>

/*MC

LineSearchShell - Provides context for a user-provided line search routine.

The user routine has one argument, the LineSearch context.  The user uses the interface to
extract line search parameters and set them accordingly when the computation is finished.

Any of the other line searches may serve as a guide to how this is to be done.

Level: advanced

 M*/

typedef struct {
  LineSearchUserFunc func;
  void               *ctx;
} LineSearch_Shell;

#undef __FUNCT__
#define __FUNCT__ "LineSearchShellSetUserFunc"
/*@C
   LineSearchShellSetUserFunc - Sets the user function for the LineSearch Shell implementation.

   Not Collective

   Level: advanced

   .keywords: LineSearch, LineSearchShell, Shell

   .seealso: LineSearchShellGetUserFunc()
@*/

PetscErrorCode LineSearchShellSetUserFunc(LineSearch linesearch, LineSearchUserFunc func, void *ctx) {

  PetscErrorCode   ierr;
  PetscBool        flg;
  LineSearch_Shell *shell = (LineSearch_Shell *)linesearch->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, LineSearch_CLASSID, 1);
  ierr = PetscTypeCompare((PetscObject)linesearch,LINESEARCHSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    shell->ctx = ctx;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "LineSearchShellGetUserFunc"
/*@C
   LineSearchShellGetUserFunc - Gets the user function and context for the shell implementation.

   Not Collective

   Level: advanced

   .keywords: LineSearch, LineSearchShell, Shell

   .seealso: LineSearchShellSetUserFunc()
@*/
PetscErrorCode LineSearchShellGetUserFunc(LineSearch linesearch, LineSearchUserFunc *func, void **ctx) {

  PetscErrorCode   ierr;
  PetscBool        flg;
  LineSearch_Shell *shell = (LineSearch_Shell *)linesearch->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(linesearch, LineSearch_CLASSID, 1);
  if (func) PetscValidPointer(func,2);
  if (ctx)  PetscValidPointer(ctx,3);
  ierr = PetscTypeCompare((PetscObject)linesearch,LINESEARCHSHELL,&flg);CHKERRQ(ierr);
  if (flg) {
    *ctx  = shell->ctx;
    *func = shell->func;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "LineSearchApply_Shell"
PetscErrorCode  LineSearchApply_Shell(LineSearch linesearch)
{
  LineSearch_Shell *shell = (LineSearch_Shell *)linesearch->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  /* apply the user function */
  if (shell->func) {
    ierr = (*shell->func)(linesearch, shell->ctx);CHKERRQ(ierr);
  } else {
    SETERRQ(((PetscObject)linesearch)->comm, PETSC_ERR_USER, "LineSearchShell needs to have a shell function set with LineSearchShellSetUserFunc");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LineSearchDestroy_Shell"
PetscErrorCode  LineSearchDestroy_Shell(LineSearch linesearch)
{
  LineSearch_Shell *shell = (LineSearch_Shell *)linesearch->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFree(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "LineSearchCreate_Shell"
PetscErrorCode LineSearchCreate_Shell(LineSearch linesearch)
{

  LineSearch_Shell     *shell;
  PetscErrorCode       ierr;

  PetscFunctionBegin;

  linesearch->ops->apply          = LineSearchApply_Shell;
  linesearch->ops->destroy        = LineSearchDestroy_Shell;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;

  ierr = PetscNewLog(linesearch, LineSearch_Shell, &shell);CHKERRQ(ierr);
  linesearch->data = (void*) shell;
  PetscFunctionReturn(0);
}
EXTERN_C_END