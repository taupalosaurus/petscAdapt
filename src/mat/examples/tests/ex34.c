#ifndef lint
static char vcid[] = "$Id: ex34.c,v 1.18 1996/04/04 21:16:53 curfman Exp $";
#endif

static char help[] = 
"Reads a matrix and vector from a file and writes to another. Input options:\n\
  -fin <input_file> : file to load.  For an example of a 5X5 5-pt. stencil,\n\
                      use the file matbinary.ex.\n\
  -fout <output_file> : file for saving output matrix and vector\n\n";

#include "mat.h"
#include <stdio.h>

int main(int argc,char **args)
{
  int        ierr,flg;
  Vec        x;
  Mat        A;
  char       file[256];
  Viewer     fd;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read matrix and RHS */
  ierr = OptionsGetString(PETSC_NULL,"-fin",file,255,&flg); CHKERRA(ierr);
  if (!flg) SETERRA(1,help);
  ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,file,BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatLoad(fd,MATSEQAIJ,&A); CHKERRA(ierr);
  ierr = VecLoad(fd,&x); CHKERRA(ierr);
  ierr = ViewerDestroy(fd); CHKERRA(ierr);

  /* Write matrix and vector */
  ierr = OptionsGetString(PETSC_NULL,"-fout",file,255,&flg); CHKERRA(ierr);
  if (!flg) SETERRA(1,help);
  ierr = ViewerFileOpenBinary(MPI_COMM_WORLD,file,BINARY_CREATE,&fd);CHKERRA(ierr);
  ierr = MatView(A,fd); CHKERRA(ierr);
  ierr = VecView(x,fd); CHKERRA(ierr);

  /* Free data structures */
  ierr = MatDestroy(A); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = ViewerDestroy(fd); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

