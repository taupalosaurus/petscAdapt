

static char help[] = "Example demonstrating color map\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  DrawCtx draw;
  int     ierr, x = 0, y = 0, width = 256, height = 256,i; 
  double  xx;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,help);

  ierr = DrawOpenX(MPI_COMM_SELF,0,"Window Title",x,y,width,height,&draw);
  CHKERR(ierr);
  for ( i=0; i<256; i++) {
    ierr = DrawLine(draw,0.0,((double)i)/256.,1.0,((double)i)/256.,i,i);
  }
  ierr = DrawFlush(draw);
  sleep(2);
  PetscFinalize();
  return 0;
}
 
