#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex3.c,v 1.5 1999/04/19 22:10:43 bsmith Exp balay $";
#endif
/*
       Tests ISAllGather()
*/

static char help[] = "Tests ISAllGather()\n\n";

#include "is.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        i, n, ierr,*indices,rank,size;
  IS         is,newis;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  /*
     Create IS
  */
  n = 4 + rank;
  indices = (int *) PetscMalloc( n*sizeof(int) );CHKPTRQ(indices);
  for ( i=0; i<n; i++ ) {
    indices[i] = rank + i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,indices,&is);CHKERRA(ierr);
  PetscFree(indices);

  /*
      Stick them together from all processors 
  */
  ierr = ISAllGather(is,&newis);CHKERRA(ierr);

  if (rank == 0) {
    ierr = ISView(newis,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  }

  ierr = ISDestroy(newis);CHKERRA(ierr);
  ierr = ISDestroy(is);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 






