// -*- c++ -*-
// $Header: /u/drspeech/repos/quicknet2/QN_filters.h,v 1.23 2011/08/16 19:42:30 davidj Exp $

#ifndef QN_filters_h_INCLUDED
#define QN_filters_h_INCLUDED

// This header contains various simple "filters" for the QuickNet
// stream objects.  Filters do things like caclulate deltas, return
// only some subset of the database or normalize features.


#include <QN_config.h>
#include <assert.h>
#include <stddef.h>
#ifdef QN_HAVE_LIMITS_H
#include <limits.h>
#endif
#include "QN_Logger.h"
#include "QN_types.h"
#include "QN_streams.h"
#include "QN_args.h"

////////////////////////////////////////////////////////////////
// A "prototype" filter.  This does nothing of itself, but is
// a useful basis for other, more sophisticated filters.

class QN_InFtrStream_ProtoFilter : public QN_InFtrStream
{
public:
    // "a_debug" is the level of debugging verbosity.
    // "a_classname" is the name of the class to use in log messages.
    // "a_dbgname" is the name of the instance to use in log messages.
    // "str" is the stream we are filtering
    QN_InFtrStream_ProtoFilter(int a_debug, const char* a_classname,
			       const char* a_dbgname,
			       QN_InFtrStream& a_str);
    ~QN_InFtrStream_ProtoFilter();

    size_t num_ftrs();
    QN_SegID nextseg();
    size_t read_ftrs(size_t cnt, float* ftrs);
    int rewind();
    
    size_t num_segs();
    size_t num_frames(size_t segno = QN_ALL);
    int get_pos(size_t* segno, size_t* frameno);
    QN_SegID set_pos(size_t segno, size_t frameno);

protected:
    QN_ClassLogger log;		// Logging object.
    QN_InFtrStream& str;	// The stream we filtering.
};

////////////////////////////////////////////////////////////////
// The "Narrow" filter selects a subset of all the features to return.  It will
// work with both random access and sequential access streams.  Note that
// taking two "narrowings" of one stream is a bad idea, as the notion of the
// current sentence and frame will get confused.

class QN_InFtrStream_Narrow : public QN_InFtrStream_ProtoFilter
{
public:
    // And enum that can be used to select all sentences
    // "a_debug" is the level of debugging verbosity.
    // "a_dbgname" is the tag used for this instance in debugging output.
    // "a_str" is the stream we are taking a subset of.
    // "a_first_ftr" is the number of the features in stream that we
    //             will consider segment 0.
    // "a_num_ftrs" is the number of featurs to return - QN_ALL means all of
    //              them.
    // "a_buf_frames" is the number of frames that will be requested from
    //                the input stream.
    QN_InFtrStream_Narrow(int a_debug, const char* a_dbgname,
			  QN_InFtrStream& a_str,
			  size_t a_first_ftr, size_t a_num_ftrs,
			  size_t a_buf_frames=32);
    ~QN_InFtrStream_Narrow();
    size_t num_ftrs();
    size_t read_ftrs(size_t a_frames, float* ftrs);

private:
    // Stream being filtered and debugging flag handled by "ProtoFilter"
    const size_t input_numftrs;	// The number of features in the input stream.
    const size_t first_ftr;	// The number of the first feature required.
    const size_t output_numftrs;	// The number of features required.
    const size_t buf_frames;	// The number of frame in the buffer.

    float* const buffer;	// A buffer for reading features into.
};

// The number of features returned in each frame of the stream.

inline size_t
QN_InFtrStream_Narrow::num_ftrs()
{
    return output_numftrs;
}

////////////////////////////////////////////////////////////////
// The "Pad" filter modifies the length of an utterance by adding
// extra frames at the start and/or end of the utterance.
// No buffer is used.

class QN_InFtrStream_Pad : public QN_InFtrStream
{
public:
    // "a_head_pad" is the number of frames to pad at start (can be negative)
    // "a_tail_pad" is the number of frames to pad at start (can be negative)
    // "a_pad_type" is the type of padding to use (QN_PAD_ZERO or QN_PAD_REPEAT)
    QN_InFtrStream_Pad(int a_debug, const char* a_dbgname,
		       QN_InFtrStream& a_str,
		       size_t a_head_pad, size_t a_tail_pad,
		       QN_PadType a_pad_type);
    ~QN_InFtrStream_Pad();
    QN_SegID nextseg(void);
    size_t read_ftrs(size_t a_frames, float* a_ftrs);
    int rewind(void);
    QN_SegID set_pos(size_t a_segno, size_t a_frameno);
    int get_pos(size_t* a_segnop, size_t* a_framenop);
    size_t num_ftrs(void);
    size_t num_segs(void);
    size_t num_frames(size_t a_segno);
  
private:
    QN_ClassLogger log;		// Logging object.
    QN_InFtrStream& str;	// The stream we filtering.
    const int head_pad;		// Number of frames to pad at start of seg
    const int tail_pad;		// Number of frames to pad at end of seg
    const QN_PadType pad_type;	// Type of padding
    float* frame_buf;		// Buffer for one frame
    float* pad_buf;		// Buffer for padding
    const size_t n_ftrs;	// Number of features in the stream
    size_t my_seg;		// Current segment
    size_t my_frameno;		// Current frame
    size_t end_marker;		// Last read frame;
};

// ////////////////////////////////////////////////////////////////
// // Padding a label stream

// class QN_InLabStream_Pad : public QN_InLabStream
// {
// public:
//   QN_InLabStream_Pad(int a_debug, const char* a_dbgname,
// 		     QN_InLabStream& a_str,
// 		     int a_pad);
//   ~QN_InLabStream_Pad(); 

//   size_t num_labs();
//   QN_SegID nextseg();
//   size_t read_labs(size_t a_cnt, QNUInt32* a_labs);
//   int rewind();

//   size_t num_segs();
//   size_t num_frames(size_t a_segno);
//   int get_pos(size_t* a_segno, size_t* a_frameno);
//   QN_SegID set_pos(size_t a_segno, size_t a_frameno);

// private:
//   QN_ClassLogger log;		// Logging object.
//   QN_InLabStream& str;	        // The stream we are filtering.
//   const int pad;
//   const size_t num_labels;
//   size_t my_seg;
//   size_t my_frameno;
//   size_t num_frames_unpadded;   // size of current utterance
//   size_t num_frames_padded;     // size of current utterance + 2*pad
//   QNUInt32* utt_buffer;         // for current utterance data + padding
//   size_t buffer_size;           // maximum size of buffer (can grow)
//   void fill_utt_buffer(void);   // read data into buffer, and pad it
// };


////////////////////////////////////////////////////////////////
// The "Norm" filter normalizes each feature in the frame by adding a
// bias and multiplying by a scaling factor. It will
// work with both random access and sequential access streams.

class QN_InFtrStream_Norm : public QN_InFtrStream_ProtoFilter
{
public:
    // Create a "norm" filter.
    // Note that copies of the bias and scale vectors are taken so the
    // original vectors can be deleted.
    // "a_debug" is the level of debugging verbosity
    // "a_dbgname" is the tag used for this instance in debugging output.
    // "a_str" is the stream we are taking a subset of
    // "a_bias_vec" is the values we will add (it must be the same length as
    //              a frame in the stream being normalized).
    // "a_scale_vec" is the values we will multiply by.
    QN_InFtrStream_Norm(int a_debug, const char* a_dbgname,
			QN_InFtrStream& a_str,
			const float* a_bias_vec, const float* a_scale_vec);
    ~QN_InFtrStream_Norm();
    size_t read_ftrs(size_t a_frames, float* ftrs);

protected:
    // Most stuff provided by "ProtoFilter" base class.
    const size_t num_ftrs;	// A local version to save having to get if
				// from the input stream every time.
    float* bias_vec;
    float* scale_vec;

};

////////////////////////////////////////////////////////////////
// The "NormUtts" filter normalizes the features in a pfile by first 
// calculating the mean and variance for each utterance, then using 
// these values to normalize the features in that utterance.  As a result, 
// it has to read through each utterance in entirety before any frames 
// can be returned.  It does this by calculating the means and sds for 
// each utterance after a set_pos or a next_seg; as such, it will only 
// work with random-access feature streams.

class QN_InFtrStream_NormUtts : public QN_InFtrStream_Norm
{
public:
    // Create a "normutts" filter.
    // "a_debug" is the level of debugging verbosity
    // "a_dbgname" is the tag used for this instance in debugging output.
    // "a_str" is the stream we are taking a subset of
    // bias and scale for the underlying Norm stream are set on-the-fly
    QN_InFtrStream_NormUtts(int a_debug, const char* a_dbgname,
			    QN_InFtrStream& a_str);
    ~QN_InFtrStream_NormUtts();

    // We only have to specialize the fns that change the seg
    QN_SegID nextseg(void);
    int rewind(void);
    QN_SegID set_pos(size_t segno, size_t frameno);

private:
    void calc_mean_sd(void);	// Calculate the mean and sd thru end of seg
                                // & write to bias/scale of parent Norm filter

    // Most stuff provided by parent class
    size_t my_seg;	// current seg no, so can tell if set_pos needs recalc
};


// ////////////////////////////////////////////////////////////////
// // The "NormSpkr" filter normalizes the feature stream by reading
// // the speaker-specific stats from a special stream, cf. qnnorm.
// // Additionally, it requires a speaker label file.
// // Upon creation, all speaker-specific stats and labels are copied
// // into memory (usually reasonably small) so the norm stream and
// // speaker label stream may be destroyed after instantiation.
// // Most of the class is patterned on QN_InFtrStream_NormUtts

// class QN_InFtrStream_NormSpkr : public QN_InFtrStream_Norm
// {
// public:
//   QN_InFtrStream_NormSpkr(int a_debug, const char* a_dbgname,
// 			  QN_InFtrStream& a_str, 
// 			  QN_InFtrStream& a_norm_str,
// 			  QN_InLabStream& a_lab_str);
//   ~QN_InFtrStream_NormSpkr();
//  
//   // We only have to specialize the fns that change the seg
//   QN_SegID nextseg(void);
//   int rewind(void);
//   QN_SegID set_pos(size_t segno, size_t frameno);
//
// private:
//   size_t my_seg;	
//   size_t num_ftrs;
//   // These data might get large
//   float* spkr_means;
//   float* spkr_sdevs;  
//   size_t* spkr_labs;
// };


////////////////////////////////////////////////////////////////
// A filter for contcatenating the frames of two feature streams to
// form a new, wider stream.

class QN_InFtrStream_JoinFtrs : public QN_InFtrStream
{
public:
    // "a_debug" is the level of debugging verbosity.
    // "a_dbgname" is the name of the instance to use in log messages.
    // "a_str1" is the first of the two streams to be concatenated.
    // "a_str2" is the second of the two streams to be concatenated.
    QN_InFtrStream_JoinFtrs(int a_debug,
			    const char* a_dbgname,
			    QN_InFtrStream& a_str1,
			    QN_InFtrStream& a_str2);
    ~QN_InFtrStream_JoinFtrs();

    size_t num_ftrs();
    QN_SegID nextseg();
    size_t read_ftrs(size_t cnt, float* ftrs);
    size_t read_ftrs(size_t cnt, float* ftrs, size_t stride);
    int rewind();
    
    size_t num_segs();
    size_t num_frames(size_t segno = QN_ALL);
    int get_pos(size_t* segno, size_t* frameno);
    QN_SegID set_pos(size_t segno, size_t frameno);

protected:
    QN_ClassLogger clog;		// Logging object.
    QN_InFtrStream& str1;	// The first stream.
    QN_InFtrStream& str2;	// The second stream.
    size_t str1_n_ftrs;		// No. of features in the first stream.
    size_t str2_n_ftrs;		// No. of features in the second stream.
};

////////////////////////////////////////////////////////////////
// A filter for taking an input label stream and forming a unary-encoded
// feature stream.

class QN_InFtrStream_OneHot : public QN_InFtrStream
{
public:
    // "a_debug" is the level of debugging verbosity.
    // "a_dbgname" is the name of the instance to use in log messages.
    // "a_labstr" is the input label stream.
    // "a_size" is the width of the resulting label stream.
    // "a_highval" is the value used for the one "high" feature.
    // "a_lowval" is the value used for the remaining "low" features.
    QN_InFtrStream_OneHot(int a_debug,
			  const char* a_dbgname,
			  QN_InLabStream& a_labstr,
			  size_t a_size,
			  size_t a_bufsize = 16,
			  float a_highval = 1.0f,
			  float a_lowval = 0.0f);
    ~QN_InFtrStream_OneHot();

    size_t num_ftrs();
    QN_SegID nextseg();
    size_t read_ftrs(size_t a_cnt, float* a_ftrs);
    size_t read_ftrs(size_t a_cnt, float* a_ftrs, size_t a_stride);
    int rewind();
    
    size_t num_segs();
    size_t num_frames(size_t segno = QN_ALL);
    int get_pos(size_t* segnop, size_t* framenop);
    QN_SegID set_pos(size_t segno, size_t frameno);

private:
    // A function to actually set the output features.
    void fill_ftrs(size_t frames, float* ftrs, size_t stride);

protected:
    QN_ClassLogger clog;		// Logging object.
    QN_InLabStream& labstr;	// The first stream.
    size_t size;		// The width of the output stream.
    size_t bufsize;		// The size of the buffer in frames.
    float highval;		// The value for the one high output.
    float lowval;		// The value for the many low outputs.
    QNUInt32* buf;		// Buffer for read labels.
};

////////////////////////////////////////////////////////////////
// The "Shorten" filter supplies only a contiguous subset of all the sentences
// in a given stream.
// It will work with both random access and sequential access streams.
// Note that taking two subsets of one stream is a bad idea, as the notion
// of the current sentence and frame will get confused - see the "cut" filters
// for the right way to do it.
//
// *** WARNING - not working yet! (davidj - 23rd June '95)


class QN_InFtrStream_Shorten : public QN_InFtrStream
{
public:
    // And enum that can be used to select all sentences
    enum { ALL_SEGS = QN_SIZET_BAD };
    // "dbg" is the level of debugging verbosity
    // "str" is the stream we are taking a subset of
    // "first_seg" is the number of the segment in stream that we
    //             will consider segment 0
    // "num_segs" is the number of segments we will read before pretending
    //            that all segments have been read
    QN_InFtrStream_Shorten(int dbg, QN_InFtrStream& str, size_t first_seg,
			  size_t num_segs);
    ~QN_InFtrStream_Shorten();
    size_t num_ftrs();
    int rewind();
    QN_SegID nextseg();
    size_t read_ftrs(size_t cnt, float* ftrs);

    size_t num_segs();
    size_t num_frames(size_t segno);
    int get_pos(size_t* segno, size_t* frameno);
    QN_SegID set_pos(size_t segno, size_t frameno);
private:
    const int debug;
    QN_InFtrStream& substr;	// The stream we are subsetting
    const size_t numftrs;	// The number of features in the stream
    const size_t firstseg;	// The segment number in "substr" we consider
				// to be segment 0
    const size_t numsegs;	// The number of segments in "substr" we
				// access
    size_t current_segno;	// The current segment number
    int indexed;		// 1 if indexed
    static const char* CLASSNAME; // Name of the class (for error
				// reporting) 
};

////////////////////////////////////////////////////////////////
// The "SelSegs" filter supplies only a subset of all the sentences
// in a given stream.
// It will work only with random access streams if segments are unsorted.
// It will work with both sequential and  random access streams if segments 
// are sorted.
// 
// This should replace QN_InFtrStream_Shorten
//
// Note that taking two subsets of one stream is a bad idea, as the notion
// of the current sentence and frame will get confused - see the "cut" filters
// for the right way to do it.

#ifdef OLD
class QN_InFtrStream_SelSegs : public QN_InFtrStream
{
public:
  // "dbg" is the level of debugging verbosity
  // "str" is the stream we are taking a subset of
  // "a_range_list" is a list of strided integer ranges (begin,end,stride)
  // "a_sorted" tells whether we should sort and uniq the ranges
  QN_InFtrStream_SelSegs(int dbg, QN_InFtrStream& str, 
			 QN_Arg_RangeInt *a_range_list, int a_sorted);
  ~QN_InFtrStream_SelSegs();

  size_t num_ftrs();
  size_t read_ftrs(size_t cnt, float* ftrs);
  int rewind();
  QN_SegID nextseg();

  size_t num_segs();
  size_t num_frames(size_t segno = QN_ALL);
  int get_pos(size_t* segno, size_t* frameno);
  QN_SegID set_pos(size_t segno, size_t frameno);
private:
  const int debug; 
  QN_InFtrStream& substr;	// The stream we are subsetting
  const size_t numftrs;	// The number of features in the stream
  size_t current_segno;	// The current segment number
  size_t numsegs;       // The number of segments
  size_t numframes;     // The number of frames in the database
  int indexed;		// 1 if indexed
  int sorted;                 // 1 if sorted
  size_t *segment_map;        // indexes into segments;
  static const char* CLASSNAME; // Name of the class (for error
                                // reporting) 
};

class QN_InLabStream_SelSegs : public QN_InLabStream
{
public:
  // "dbg" is the level of debugging verbosity
  // "str" is the stream we are taking a subset of
  // "a_range_list" is a list of strided integer ranges (begin,end,stride)
  // "a_sorted" tells whether we should sort and uniq the ranges
  QN_InLabStream_SelSegs(int dbg, QN_InLabStream& str, 
			 QN_Arg_RangeInt *a_range_list, int a_sorted);
  ~QN_InLabStream_SelSegs(); 

  size_t num_labs();
  size_t read_labs(size_t cnt, QNUInt32* labs);
  int rewind();
  QN_SegID nextseg();

  size_t num_segs();
  size_t num_frames(size_t segno = QN_ALL);
  int get_pos(size_t* segno, size_t* frameno);
  QN_SegID set_pos(size_t segno, size_t frameno);
private:
  const int debug; 
  QN_InLabStream& substr;	// The stream we are subsetting
  const size_t numlabs;	// The number of features in the stream
  size_t current_segno;	// The current segment number
  size_t numsegs;       // The number of segments
  size_t numframes;     // The number of frames in the database
  int indexed;		// 1 if indexed
  int sorted;                 // 1 if sorted
  size_t *segment_map;        // indexes into segments;
  static const char* CLASSNAME; // Name of the class (for error
				// reporting) 
};

#endif
////////////////////////////////////////////////////////////////
// This is a useful stream for generating label information.
// It produces a stream containing three labels, the first of which is
// the epoch (rewind increases this), the second is the segment number
// of that segment, the third the frame number of that
// frame.  The number of segments and the number of frames in each
// segment can be either random (useful for testing) or based on an existing
// input stream (useful for keeping track of where stuff comes from after
// complicated windowing, randomizaiton etc.)
////////////////////////////////////////////////////////////////

class QN_InLabStream_FrameInfo : public QN_InLabStream
{
public:
    // Offsets into each label vector.
    enum {
	EPOCH_OFFSET = 0,
	SEGNO_OFFSET = 1,
	FRAMENO_OFFSET = 2,
	NUM_LABELS = 3
    };
public:
    // Constructor for random length sentences.
    // "a_debug" is the level of debugging verbosity.
    // "a_dbgname" is the tag used for this instance in debugging output.
    // "a_min_segs" is the minimum allowed number of segments - cannot be 0.
    // "a_max_segs" is the maximum allowed number of segments.
    // "a_min_frames" is the minimum allowed number of frames in one segment.
    // "a_max_frames" is the maximum allowed number of frames in one segment.
    // "a_seed" is a seed for the random number generator.
    QN_InLabStream_FrameInfo(int a_debug, const char* a_dbgname,
			     size_t a_min_segs, size_t a_max_segs,
			     size_t a_min_frames, size_t a_max_frames,
			     int a_seed);
    // Constructor for sentence lengths based on an existing stream.
    // "a_debug" is the level of debugging verbosity
    // "a_dbgname" is the tag used for this instance in debugging output.
    // "a_str" is the stream we are taking a subset of
    QN_InLabStream_FrameInfo(int a_debug, const char* a_dbgname,
			     QN_InStream& str);
    ~QN_InLabStream_FrameInfo();
    size_t num_labs() { return NUM_LABELS; };
    int rewind();
    QN_SegID nextseg();
    size_t read_labs(size_t a_cnt, QNUInt32* a_labs);

    size_t num_segs();
    size_t num_frames(size_t segno = QN_ALL);
    int get_pos(size_t* a_segnop, size_t* a_framenop);
    QN_SegID set_pos(size_t a_segno, size_t a_frameno);

private:
    QN_ClassLogger clog;	// Logging object.

    size_t n_segs;		// The number of segments in the stream.
    size_t* seglens;		// An array of sentence lengths.
    size_t total_frames;	// The total number of frames in the stream.

    size_t epoch;		// The current epoch.
    size_t segno;		// The current segment number.
    size_t frameno;		// The current frame number.
};

////////////////////////////////////////////////////////////////

inline size_t
QN_InFtrStream_ProtoFilter::num_ftrs()
{
    return str.num_ftrs();
}

inline int
QN_InFtrStream_ProtoFilter::rewind()
{
    return str.rewind();
}

inline QN_SegID 
QN_InFtrStream_ProtoFilter::nextseg()
{
    return str.nextseg();
}

inline size_t
QN_InFtrStream_ProtoFilter::read_ftrs(size_t cnt, float* ftrs)
{
    return str.read_ftrs(cnt, ftrs);
}

inline size_t
QN_InFtrStream_ProtoFilter::num_segs()
{
    return str.num_segs();
}

inline size_t
QN_InFtrStream_ProtoFilter::num_frames(size_t segno)
{
    return str.num_frames(segno);
}

inline int
QN_InFtrStream_ProtoFilter::get_pos(size_t* segno, size_t* frameno)
{
    return str.get_pos(segno, frameno); // Call substream get_pos
}

inline QN_SegID
QN_InFtrStream_ProtoFilter::set_pos(size_t segno, size_t frameno)
{
    return str.set_pos(segno, frameno);
}


////////////////////////////////////////////////////////////////

inline size_t
QN_InFtrStream_JoinFtrs::num_ftrs()
{
    return str1_n_ftrs + str2_n_ftrs;
}

// This is just a call of the strided version.
inline size_t
QN_InFtrStream_JoinFtrs::read_ftrs(size_t cnt, float* ftrs)
{
    return read_ftrs(cnt, ftrs, QN_ALL);
}

////////////////////////////////////////////////////////////////

inline size_t
QN_InFtrStream_OneHot::num_ftrs()
{
    return size;
}

inline QN_SegID
QN_InFtrStream_OneHot::nextseg()
{
    return labstr.nextseg();
}

// This is just a call of the strided version.
inline size_t
QN_InFtrStream_OneHot::read_ftrs(size_t cnt, float* ftrs)
{
    return read_ftrs(cnt, ftrs, QN_ALL);
}

inline int
QN_InFtrStream_OneHot::rewind()
{
    return labstr.rewind();
}
    
inline size_t
QN_InFtrStream_OneHot::num_segs()
{
    return labstr.num_segs();
}

inline size_t
QN_InFtrStream_OneHot::num_frames(size_t segno)
{
    return labstr.num_frames(segno);
}


inline int
QN_InFtrStream_OneHot::get_pos(size_t* segnop, size_t* framenop)
{
    return labstr.get_pos(segnop, framenop);
}

inline QN_SegID
QN_InFtrStream_OneHot::set_pos(size_t segno, size_t frameno)
{
    return labstr.set_pos(segno, frameno);
}

////////////////////////////////////////////////////////////////

#endif // #ifdef QN_filters_h_INCLUDED
