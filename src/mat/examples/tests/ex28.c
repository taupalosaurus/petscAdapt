#ifndef lint
static char vcid[] = "$Id: ex28.c,v 1.11 1996/08/01 14:34:00 balay Exp $";
#endif

static char help[] = "Tests MatReorderForNonzeroDiagonal()\n\n";

#include "mat.h"
#include <stdio.h>

int main(int argc, char **args)
{
  Mat    A,LU;
  Vec    x,y;
  int    nnz[4]={2,1,1,1},col[4],i,ierr;
  Scalar values[4];
  IS     rowperm, colperm;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreateSeqAIJ(MPI_COMM_WORLD,4,4,2,nnz,&A); CHKERRA(ierr);

  /* build test matrix */
  values[0]=1.0;values[1]=-1.0;
  col[0]=0;col[1]=2; i=0;
  ierr = MatSetValues(A,1,&i,2,col,values,INSERT_VALUES); CHKERRA(ierr);
  values[0]=1.0;
  col[0]=1;i=1;
  ierr = MatSetValues(A,1,&i,1,col,values,INSERT_VALUES); CHKERRA(ierr);
  values[0]=-1.0;
  col[0]=3;i=2;
  ierr = MatSetValues(A,1,&i,1,col,values,INSERT_VALUES); CHKERRA(ierr);
  values[0]=1.0;
  col[0]=2;i=3;
  ierr = MatSetValues(A,1,&i,1,col,values,INSERT_VALUES); CHKERRA(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_SELF); CHKERRA(ierr);

  ierr = MatGetReordering(A,ORDER_NATURAL,&rowperm,&colperm); CHKERRA(ierr);
  ierr = MatReorderForNonzeroDiagonal(A,1.e-12,rowperm,colperm); CHKERRA(ierr);
  PetscPrintf(MPI_COMM_SELF,"column and row perms\n");
  ierr = ISView(rowperm,0); CHKERRA(ierr);
  ierr = ISView(colperm,0); CHKERRA(ierr);
  ierr = MatLUFactorSymbolic(A,rowperm,colperm,1.0,&LU); CHKERRA(ierr);
  ierr = MatLUFactorNumeric(A,&LU); CHKERRA(ierr);
  ierr = MatView(LU,VIEWER_STDOUT_SELF); CHKERRA(ierr);
  ierr = VecCreate(MPI_COMM_WORLD,4,&x); CHKERRA(ierr);
  ierr = VecCreate(MPI_COMM_WORLD,4,&y); CHKERRA(ierr);
  values[0]=0;values[1]=1.0;values[2]=-1.0;values[3]=1.0;
  for (i=0; i<4; i++) col[i]=i;
  ierr = VecSetValues(x,4,col,values,INSERT_VALUES); CHKERRA(ierr);
  ierr = VecAssemblyBegin(x); CHKERRA(ierr);
  ierr = VecAssemblyEnd(x); CHKERRA(ierr);
  ierr = VecView(x,VIEWER_STDOUT_SELF); CHKERRA(ierr);

  ierr = MatSolve(LU,x,y); CHKERRA(ierr);
  ierr = VecView(y,VIEWER_STDOUT_SELF); CHKERRA(ierr);

  ierr = ISDestroy(rowperm); CHKERRA(ierr);
  ierr = ISDestroy(colperm); CHKERRA(ierr);
  ierr = MatDestroy(LU); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}


