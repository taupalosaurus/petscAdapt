#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.14 1999/05/12 03:24:58 bsmith Exp balay $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H
 
#define PARCH_IRIX

#define PETSC_HAVE_LIMITS_H
#define PETSC_HAVE_PWD_H 
#define PETSC_HAVE_STRING_H 
#define PETSC_HAVE_STROPTS_H
#define PETSC_HAVE_MALLOC_H 
#define PETSC_HAVE_X11 
#define PETSC_HAVE_FORTRAN_UNDERSCORE 
#define PETSC_HAVE_DRAND48  
#define PETSC_HAVE_GETDOMAINNAME 
#define PETSC_HAVE_UNAME 
#define PETSC_HAVE_UNISTD_H 
#define PETSC_HAVE_SYS_TIME_H

#define PETSC_HAVE_FORTRAN_UNDERSCORE 

#define PETSC_HAVE_DOUBLE_ALIGN
#define PETSC_HAVE_DOUBLE_ALIGN_MALLOC

#define PETSC_HAVE_MEMALIGN

#define PETSC_USE_DBX_DEBUGGER
#define PETSC_HAVE_SYS_RESOURCE_H

#define PETSC_SIZEOF_VOIDP 4
#define PETSC_SIZEOF_INT 4
#define PETSC_SIZEOF_DOUBLE 8

#define PETSC_WORDS_BIGENDIAN 1

#define PETSC_HAVE_4ARG_SIGNAL_HANDLER

#endif
