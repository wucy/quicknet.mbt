/*
** $Header: /u/drspeech/repos/quicknet2/QN_config_tail.h,v 1.3 2011/09/06 21:20:12 davidj Exp $
** 
** QN_config_tail.h - stuff to add onto the end of QN_config.h
*/

#if defined(QN__FILE_OFFSET_BITS) && !defined(_FILE_OFFSET_BITS)
#define _FILE_OFFSET_BITS QN__FILE_OFFSET_BITS
#endif
#if defined(QN__LARGEFILE_SOURCE) && !defined(_LARGEFILE_SOURCE)
#define _LARGEFILE_SOURCE QN__LARGEFILE_SOURCE
#endif
#if defined(QN__LARGE_FILES) && !defined(_LARGE_FILES)
#define _LARGE_FILES QN__LARGE_FILES
#endif

/* Deal with the va_copy mess */

#ifndef va_copy
#ifdef __va_copy
#define va_copy(dst,src) __va_copy(dst,src)
#else
#define va_copy(dst,src) dst=src
#endif
#endif
