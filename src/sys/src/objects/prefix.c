#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: prefix.c,v 1.18 1999/04/21 20:43:25 bsmith Exp balay $";
#endif
/*
     Provides utility routines for manulating any type of PETSc object.
*/
#include "petsc.h"  /*I   "petsc.h"    I*/

#undef __FUNC__  
#define __FUNC__ "PetscObjectSetOptionsPrefix"
/*
   PetscObjectSetOptionsPrefix - Sets the prefix used for searching for all 
   options of PetscObjectType in the database. 

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
.  prefix - the prefix string to prepend to option requests of the object.

   Notes: 
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

.keywords: object, set, options, prefix, database
*/
int PetscObjectSetOptionsPrefix(PetscObject obj,const char prefix[])
{
  int ierr;

  PetscFunctionBegin;
  if (obj->prefix) PetscFree(obj->prefix);
  if (prefix == PETSC_NULL) {obj->prefix = PETSC_NULL; PetscFunctionReturn(0);}
  if (prefix[0] == '-') SETERRQ(PETSC_ERR_ARG_WRONG,1,"Options prefix should not begin with a hypen");

  obj->prefix = (char*) PetscMalloc((1+PetscStrlen(prefix))*sizeof(char));CHKPTRQ(obj->prefix);
  ierr = PetscStrcpy(obj->prefix,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectAppendOptionsPrefix"
/*
   PetscObjectAppendOptionsPrefix - Sets the prefix used for searching for all 
   options of PetscObjectType in the database. 

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
.  prefix - the prefix string to prepend to option requests of the object.

   Notes: 
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

.keywords: object, append, options, prefix, database
*/
int PetscObjectAppendOptionsPrefix(PetscObject obj,const char prefix[])
{
  char *buf = obj->prefix ;
  int  ierr;

  PetscFunctionBegin;
  if (!prefix) {PetscFunctionReturn(0);}
  if (!buf) {
    ierr = PetscObjectSetOptionsPrefix(obj, prefix);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (prefix[0] == '-') SETERRQ(PETSC_ERR_ARG_WRONG,1,"Options prefix should not begin with a hypen");

  obj->prefix = (char*)PetscMalloc((1+PetscStrlen(prefix)+PetscStrlen(buf))*sizeof(char));CHKPTRQ(obj->prefix);
  ierr = PetscStrcpy(obj->prefix,buf);CHKERRQ(ierr);
  ierr = PetscStrcat(obj->prefix,prefix);CHKERRQ(ierr);
  ierr = PetscFree(buf);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectGetOptionsPrefix"
/*
   PetscObjectGetOptionsPrefix - Gets the prefix of the PetscObject.

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

.keywords: object, get, options, prefix, database
*/
int PetscObjectGetOptionsPrefix(PetscObject obj,char *prefix[])
{
  PetscFunctionBegin;
  *prefix = obj->prefix;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscObjectPrependOptionsPrefix"
/*
   PetscObjectPrependOptionsPrefix - Sets the prefix used for searching for all 
   options of PetscObjectType in the database. 

   Input Parameters:
.  obj - any PETSc object, for example a Vec, Mat or KSP.
.  prefix - the prefix string to prepend to option requests of the object.

   Notes: 
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

.keywords: object, append, options, prefix, database
*/
int PetscObjectPrependOptionsPrefix(PetscObject obj,const char prefix[])
{
  char *buf = obj->prefix ;
  int  ierr;

  PetscFunctionBegin;
  if (!prefix) {PetscFunctionReturn(0);}
  if (!buf) {
    ierr = PetscObjectSetOptionsPrefix(obj, prefix);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (prefix[0] == '-') SETERRQ(PETSC_ERR_ARG_WRONG,1,"Options prefix should not begin with a hypen");

  obj->prefix = (char*)PetscMalloc((1+PetscStrlen(prefix)+PetscStrlen(buf))*sizeof(char));CHKPTRQ(obj->prefix);
  ierr = PetscStrcpy(obj->prefix,prefix);CHKERRQ(ierr);
  ierr = PetscStrcat(obj->prefix,buf);CHKERRQ(ierr);
  PetscFree(buf);
  PetscFunctionReturn(0);
}

