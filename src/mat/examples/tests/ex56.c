#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex56.c,v 1.19 1999/03/19 21:19:59 bsmith Exp balay $";
#endif
static char help[] = "Test the use of MatSetValuesBlocked(), MatZeroRows() for \n\
rectangular MatBAIJ matrix";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         A;
  int         bs=3,m=4,n=6,i,j,val = 10,ierr,flg,size,rank,rstart;
  Scalar      x[6][9],y[3][3],one=1.0;
  int         row[2],col[3],eval;
  IS          is;
  PetscInitialize(&argc,&args,(char *)0,help);

  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  
  if (size == 1) {
    ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,bs,m*bs,n*bs,1,PETSC_NULL,&A);CHKERRA(ierr);
  } else {
    ierr = MatCreateMPIBAIJ(PETSC_COMM_WORLD,bs,m*bs,n*bs,PETSC_DECIDE,PETSC_DECIDE,1,
                            PETSC_NULL,1,PETSC_NULL,&A);CHKERRA(ierr);
  }

  ierr = OptionsHasName(PETSC_NULL,"-column_oriented",&flg);CHKERRA(ierr);
  if (flg) { 
    ierr = MatSetOption(A,MAT_COLUMN_ORIENTED);CHKERRA(ierr); 
    eval = 6;
  } else {
    eval = 9;
  }

  ierr = OptionsHasName(PETSC_NULL,"-ass_extern",&flg);CHKERRA(ierr);
  if (flg && (size != 1))    rstart = m*((rank+1)%size);
  else                       rstart = m*(rank);

  row[0] =rstart+0;  row[1] =rstart+2;
  col[0] =rstart+0;  col[1] =rstart+1;  col[2] =rstart+3;
  for (i=0; i<6; i++) {
    for (j =0; j< 9; j++ ) x[i][j] = (Scalar)val++;
  }

  ierr = MatSetValuesBlocked(A,2,row,3,col,&x[0][0],INSERT_VALUES);CHKERRA(ierr);



  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /*
  This option does not work for rectangular matrices
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR);CHKERRA(ierr);
  */
  
  ierr = MatSetValuesBlocked(A,2,row,3,col,&x[0][0],INSERT_VALUES);CHKERRA(ierr);

  /* Do another MatSetValues to test the case when only one local block is specified */
  for (i=0; i<3; i++) {
    for (j =0; j<3 ; j++ )  y[i][j] = (Scalar)(10 + i*eval + j);
  }
  ierr = MatSetValuesBlocked(A,1,row,1,col,&y[0][0],INSERT_VALUES);CHKERRA(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  
  ierr = OptionsHasName(PETSC_NULL,"-zero_rows",&flg);CHKERRA(ierr);
  if (flg) {
    col[0] = rstart*bs+0;
    col[1] = rstart*bs+1;
    col[2] = rstart*bs+2;
    ierr = ISCreateGeneral(MPI_COMM_SELF,3,col,&is);CHKERRA(ierr);
    ierr = MatZeroRows(A,is,&one);CHKERRA(ierr);
    ISDestroy(is);
  }

  ierr = MatView(A,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  MatDestroy(A);
  PetscFinalize();
  return 0;
}
