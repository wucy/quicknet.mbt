// -*- C++ -*-
// $Header: $

// QN_ListStreamHTK.cc

#include <stdio.h>
#include "QN_ListStreamHTK.h"
#include "QN_utils.h"

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
// QN_InFtrStream_ListHTK - class for a list of HTK feat files
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
QN_InFtrStream_ListHTK::QN_InFtrStream_ListHTK(int a_debug, 
					       const char* a_dbgname, 
					       FILE* a_file,
					       int a_indexed)
  : QN_InFtrStream_List(a_debug, a_dbgname, a_file, a_indexed),
    feat_stream(NULL),
    fp(NULL)
{
  init();			// do base class initialization
}

size_t
QN_InFtrStream_ListHTK::read_ftrs(size_t count, float* ftrs)
{
    size_t cnt = 0;

    if(feat_stream != NULL){
	cnt = feat_stream->read_ftrs(count,ftrs);
	current_frame += cnt;
    } else {
	log.error("Attempting to read features before reading "
		  "the header.");
    }
    return cnt;
}

int
QN_InFtrStream_ListHTK::read_header()
{
  if(feat_stream != NULL) {	// remove existing feature stream
    delete feat_stream;
  }
  if(fp != NULL)		// close file pointer
    QN_close(fp);

  log.log(QN_LOG_PER_SENT, "Opening HTK file %s", 
	  curr_seg_name);
  fp = QN_open(curr_seg_name,"r");
  if(fp == NULL){
    log.error("Error opening HTK feature file: '%s'",curr_seg_name);
  }

  feat_stream = new QN_InFtrStream_HTK(debug,dbgname,fp,indexed);
  frames_this_seg = feat_stream->num_frames();
  frame_bytes = feat_stream->num_ftrs() * sizeof(float);
  feat_stream->nextseg();	// prepare for reading the file

  return 1;
}

QN_InFtrStream_ListHTK::~QN_InFtrStream_ListHTK() {
  if(feat_stream != NULL) {	// remove existing feature stream
    delete feat_stream;
  }
  if(fp != NULL)		// close file pointer
    fclose(fp);
}
