#ifdef PETSC_RCS_HEADER
"$Id: petscconf.h,v 1.19 1999/06/30 22:48:03 bsmith Exp balay $"
"Defines the configuration for this machine"
#endif

#if !defined(INCLUDED_PETSCCONF_H)
#define INCLUDED_PETSCCONF_H

#define PARCH_rs6000
#define PETSC_USE_READ_REAL_TIME

#define PETSC_HAVE_LIMITS_H
#define PETSC_HAVE_STROPTS_H 
#define PETSC_HAVE_SEARCH_H 
#define PETSC_HAVE_PWD_H 
#define PETSC_HAVE_STDLIB_H
#define PETSC_HAVE_STRING_H 
#define PETSC_HAVE_STRINGS_H 
#define PETSC_HAVE_MALLOC_H 
#define PETSC_HAVE_X11 
#define _POSIX_SOURCE 
#define PETSC_HAVE_DRAND48  
#define PETSC_HAVE_GETDOMAINNAME 
#if !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 
#endif
#define PETSC_HAVE_UNISTD_H 
#define PETSC_HAVE_SYS_TIME_H 
#define PETSC_HAVE_UNAME 
#if !defined(_XOPEN_SOURCE_EXTENDED)
#define _XOPEN_SOURCE_EXTENDED 1
#endif
#define _ALL_SOURCE   
#define PETSC_HAVE_BROKEN_REQUEST_FREE 
#define PETSC_HAVE_STRINGS_H
#define PETSC_HAVE_DOUBLE_ALIGN_MALLOC

#define PETSC_HAVE_XLF90

#define PETSC_PREFER_BZERO

#define PETSC_HAVE_READLINK
#define PETSC_HAVE_MEMMOVE

#define PETSC_HAVE_PRAGMA_DISJOINT

#define PETSC_USE_DBX_DEBUGGER
#define PETSC_HAVE_SYS_RESOURCE_H
#define PETSC_SIZEOF_VOIDP 4
#define PETSC_SIZEOF_INT 4
#define PETSC_SIZEOF_DOUBLE 8

#define PETSC_WORDS_BIGENDIAN 1
#define PETSC_NEED_SOCKET_PROTO
#define PETSC_HAVE_ACCEPT_SIZE_T

#define PETSC_HAVE_SYS_UTSNAME_H
#endif
