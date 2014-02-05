#!/bin/sh
#
# $Header: /u/drspeech/repos/quicknet2/QN3Config.sh.in,v 1.2 2010/10/29 18:20:26 davidj Exp $
#
# This file freezes some of the configuration results of QuickNet3
# so that they can be reused by subsequent QuickNet3 clients. 
# Specifically, we want to remember whether we need extra system
# libs on this platform.  We might as well define the path 
# to the installed header files too.  This sits alongside
# the libquicknet.a library file.  It is used as is in the build directory,
# but edited to use the correct directories when installed.
#

prefix='/usr/local'
exec_prefix='${prefix}'

# String to pass to linker to pick up the QuickNet library from its
# installed directory, along with any other required libraries.
QN_BUILD_LIB_SPEC="-L/home/mifs/cw564/src/quicknet.mbt -lquicknet3 -lpthread -lm -lcublas -lcudart "
QN_INSTALL_LIB_SPEC="-L${libdir} -lquicknet3 -lpthread -lm -lcublas -lcudart "
QN_LIB_SPEC=${QN_INSTALL_LIB_SPEC}

# Location of the installed include headers directory fltvec.
QN_BUILD_INC_SPEC="-I/home/mifs/cw564/src/quicknet.mbt -I/home/mifs/cw564/src/quicknet.mbt"
QN_INSTALL_INC_SPEC="-I${exec_prefix}/include/quicknet3"
QN_INC_SPEC=${QN_INSTALL_INC_SPEC}

# If a client is going to use our library, it had better use the 
# same C++ compiler as us
QN_CC='gcc'
QN_CXX='g++'
QN_CPPFLAGS=''
QN_CFLAGS='-g -O2'
QN_CXXFLAGS='-g -O2'
QN_LD_FLAGS='-L/usr/local/cuda-5.5/lib -Wl,-rpath=/usr/local/cuda-5.5/lib -L/usr/local/cuda-5.5/lib64 -Wl,-rpath=/usr/local/cuda-5.5/lib64 '


