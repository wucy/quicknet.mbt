const char* QN_MLP_ThreadFlVar_rcsid =
    "$Header: /u/drspeech/repos/quicknet2/QN_MLP_ThreadFlVar.cc,v 1.6 2011/08/16 19:39:17 davidj Exp $";

/* Must include the config.h file first */
#include <QN_config.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include "QN_types.h"
#include "QN_Logger.h"
#include "QN_MLP_ThreadFlVar.h"
#include "QN_fltvec.h"
#include "QN_intvec.h"



#ifdef QN_HAVE_LIBPTHREAD

// The actual thread worker routine, defined below
extern "C" {
	static void* worker_wrapper(void*);
};

//cz277 - dnn
QN_MLP_ThreadFlVar::QN_MLP_ThreadFlVar(int a_bp_num_layer,	//cz277 - nn fea bp
				       int a_debug,
                                       const char* a_dbgname,
                                       size_t a_n_layers,
                                       const size_t a_layer_units[QN_MLP_MAX_LAYERS],
                                       enum QN_CriterionType a_criteriontype,     //cz277 - criteria
				       enum QN_LayerType *a_layertype,     //cz277 - nonlinearity, mul actv
				       enum QN_OutputLayerType a_outtype,
                                       size_t a_size_bunch,
                                       size_t a_threads)
    : QN_MLP_BaseFl(a_debug, a_dbgname, "QN_MLP_ThreadFlVar",
                    a_size_bunch, a_n_layers, a_layer_units),	//cz277 - dnn
      criterion_type(a_criteriontype),  //cz277 - criteria
      hiddenlayer_types(a_layertype),    //cz277 - nonlinearity, mul actv
      bp_num_layer(a_bp_num_layer),	//cz277 - nn fea bp
      out_layer_type(a_outtype),
      num_threads(a_threads)
{
    int ec;
    size_t i;

    //cz277 - criteria
    switch(criterion_type)
    {
    case QN_CRITERION_QUADRATIC:
    case QN_CRITERION_XENTROPY:
            break;
    default:
	    assert(0);
    }

    //cz277 - nonlinearity, mul actv
    /*switch(hiddenlayer_type)
    {
    case QN_LAYER_LINEAR:
    case QN_LAYER_SIGMOID:
    case QN_LAYER_SOFTMAX:
    case QN_LAYER_TANH:
    case QN_LAYER_SOFTSIGN:
            break;
    default:
	    assert(0);
    }*/

    // Maybe we do not support all output layer types
    switch(out_layer_type)
    {
	case QN_OUTPUT_SIGMOID:
	//case QN_OUTPUT_SIGMOID_XENTROPY:  //cz277 - nonlinearity
	case QN_OUTPUT_TANH:
	case QN_OUTPUT_SOFTMAX:
	case QN_OUTPUT_LINEAR:
	case QN_OUTPUT_SOFTSIGN:    //cz277 - nonlinearity
		break;
	default:
		assert(0);
    }

    // Create arrays of pointers to thread-specific weights and biases
    assert(num_threads>0);
    per_thread = new PerThread[num_threads];
    for (i=0; i<num_threads; i++)
    {
	size_t j;

	for (j=0; j<MAX_LAYERS; j++)
	{
	    per_thread[i].delta_biases[j] = NULL;
	}
	for (j=0; j<MAX_WEIGHTMATS; j++)
	{
	    per_thread[i].delta_weights[j] = NULL;
	}
    }
    threadp = new float* [num_threads];
    for (i = 0; i<num_threads; i++)
	threadp[i] = NULL;

    // Some temp work space.
    // Cannot have more threads than bunch size.
    assert(num_threads<=size_bunch);
    threads = new pthread_t[num_threads];

    // Set up the "action" variable/mutex/cv worker thread command channel
    action_seq = 0;
    ec = pthread_mutex_init(&action_mutex, NULL);
    if (ec)
        clog.error("failed to init action_mutex");
    ec = pthread_cond_init(&action_cv, NULL);
    if (ec)
        clog.error("failed to init action_cv");
    done_count = 0;
    ec = pthread_mutex_init(&done_mutex, NULL);
    if (ec)
        clog.error("failed to init done_mutex");
    ec = pthread_cond_init(&done_cv, NULL);
    if (ec)
        clog.error("failed to init done_cv");

    // Create threads, make them joinable
    pthread_attr_t attr;
    ec = pthread_attr_init(&attr);
    assert(ec==0);
    ec = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    assert(ec==0);
    worker_args = new QN_MLP_ThreadFlVar_WorkerArg[num_threads];
    for (i=0; i<num_threads; i++)
    {
	worker_args[i].threadno = i;
	worker_args[i].mlp = this;
	clog.log(QN_LOG_PER_RUN,"Creating thread %lu.", i);
	ec = pthread_create(&threads[i], &attr, worker_wrapper,
			    (void*) &worker_args[i]);
	if (ec)
	{
	    clog.error("failed to create thread number %lu - %s.",
		       i, strerror(ec));
	}
    }

    clog.log(QN_LOG_PER_RUN, "Created net with %lu layers, bunchsize %lu, "
	     "using %lu threads.", n_layers, size_bunch, num_threads);
    for (i=0; i<n_layers; i++)
    {
	clog.log(QN_LOG_PER_RUN, "Layer %lu has %lu units.",
		 i+1, layer_units[i]);
    }
}

QN_MLP_ThreadFlVar::~QN_MLP_ThreadFlVar()
{
    int ec;

    clog.log(QN_LOG_PER_RUN,"Terminating threads.");
    tell_workers("destructor", ACTION_EXIT);
    wait_on_workers("destructor");

    // Wait for all of the worker threads to die
    size_t i;
    for (i = 0; i<num_threads; i++)
    {
	clog.log(QN_LOG_PER_RUN,"Waiting for end of thread %lu.", i);
        ec = pthread_join(threads[i], NULL);
	if (ec)
	    clog.error("failed to join thread %lu - %s.", i, strerror(ec));
    }
    delete[] worker_args;

    // Kill thead-related variables
    ec = pthread_cond_destroy(&done_cv);
    if (ec)
        clog.error("failed to destroy done_cv.");
    ec = pthread_mutex_destroy(&done_mutex);
    if (ec)
        clog.error("failed to destroy done_mutex.");
    ec = pthread_cond_destroy(&action_cv);
    if (ec)
        clog.error("failed to destroy action_cv.");
    ec = pthread_mutex_destroy(&action_mutex);
    if (ec)
        clog.error("failed to destroy action_mutex.");

    delete [] threads;
    delete [] threadp;
    delete [] per_thread;
}


void
QN_MLP_ThreadFlVar::forward_bunch(size_t n_frames, const float* in, float* out, const float * * wgt, const size_t n)
{
    clog.log(QN_LOG_PER_BUNCH, "forward_bunch sending ACTION_FORWARD.");
    action_n_frames = n_frames;
    action_in = in;
    action_out = out;
    tell_workers("forward_bunch", ACTION_FORWARD);
    // ....workers work...
    wait_on_workers("forward_bunch");
    clog.log(QN_LOG_PER_BUNCH, "forward_bunch workers claim they are done.");
}

void
QN_MLP_ThreadFlVar::train_bunch(size_t n_frames, const float *in,
			     const float* target, float* out, const float * * wgt, const size_t n)
{

    clog.log(QN_LOG_PER_BUNCH, "train_bunch sending ACTION_TRAIN.");
    action_n_frames = n_frames;
    action_in = in;
    action_out = out;
    action_target = target;
    tell_workers("train_bunch", ACTION_TRAIN);
    // ....workers work...
    wait_on_workers("train_bunch");
    clog.log(QN_LOG_PER_BUNCH, "train_bunch workers claim they are done.");

    // He we merge all of the deltas.
    // NOTE: The loop ordering is optimized for cache efficiency.
    size_t thread;		// The current thread.
    size_t layer;		// The current layer.
    size_t section;		// The current section.
    size_t i;			// Counter.
    float* accp;		// Pointer to weight/bias being updated.
    float sum;			// Temporary sum.
    size_t frames_per_thread;	// Number of frames per thread
    size_t useful_threads;	// Number of threads that actually did work

    //cz277 - momentum
    float *accp_delta;

    //cz277 - nn fea bp
    int bp_num_layer_idx = this->bp_num_layer;

    frames_per_thread = (size_bunch + num_threads - 1) / num_threads;
    useful_threads = (n_frames + frames_per_thread - 1) / frames_per_thread;
    for (layer = n_layers-1; layer>0 && (bp_num_layer_idx != 0); layer--, --bp_num_layer_idx)	//cz277 - nn fea bp
    {
	size_t weightmat;
	float this_neg_bias_learnrate;
	float this_neg_weight_learnrate;

	weightmat = layer - 1;
	this_neg_bias_learnrate = neg_bias_learnrate[layer];
	this_neg_weight_learnrate = neg_weight_learnrate[weightmat];
	if (this_neg_bias_learnrate != 0.0f)
	{
	    for (thread = 0 ; thread<useful_threads; thread++)
		threadp[thread] = per_thread[thread].delta_biases[layer];
	   
	    //cz277 - revisit momentum
	    accp_delta = bias_delta[layer];	

	    accp = layer_bias[layer];
	    for (i = 0; i<layer_units[layer]; i++)
	    {
		sum = 0.0f;
		for (thread = 0; thread<useful_threads; thread++)
		{
		    sum += *threadp[thread];
		    threadp[thread] += 1;
		}

		//cz277 - revisit momentum
		*accp_delta *= alpha_momentum;
		*accp_delta += this_neg_bias_learnrate * (sum + weight_decay_factor * (*accp));	//cz277 - weight decay
		(*accp++) += (*accp_delta);
		++ accp_delta;
	    }
	}
	if (this_neg_weight_learnrate != 0.0f)
	{
	    for (thread =0; thread<useful_threads; thread++)
		threadp[thread] = per_thread[thread].delta_weights[weightmat];
	    
	    //cz277 - momentum
	    accp_delta = weights_delta[weightmat];

	    accp = weights[weightmat];
	    for (i = 0; i<weights_size[weightmat]; i++)
	    {
		sum = 0.0f;
		for (thread = 0; thread<useful_threads; thread++)
		{
		    sum += *threadp[thread];
		    threadp[thread]++;
		}

		//cz277 - momentum
		*accp_delta *= alpha_momentum;
		*accp_delta += this_neg_bias_learnrate * (sum + weight_decay_factor * (*accp));	//cz277 - revisit momentum, weight decay
		(*accp++) += (*accp_delta);
		++ accp_delta;
	    }
	}
    }
}


void
QN_MLP_ThreadFlVar::tell_workers(const char* logstr, enum Action action)
{
    int ec;

    // Send an action to all of the worker threads
    ec = pthread_mutex_lock(&action_mutex);
    if (ec)
    {
        clog.error("%s - failed to lock action_mutex in tell_workers - %s",
		   logstr, strerror(ec));
    }
    action_command = action;
    action_seq++;		// Increment sequence to indicate new command
    ec = pthread_cond_broadcast(&action_cv);
    if (ec)
    {
        clog.error("%s - failed to unlock action_mutex in tell_workers - %s",
		   logstr, strerror(ec));
    }
    ec = pthread_mutex_unlock(&action_mutex);
    if (ec)
    {
        clog.error("%s - failed to unlock action_mutex in tell_workers - %s",
		   logstr, strerror(ec));
    }
}

void
QN_MLP_ThreadFlVar::wait_on_workers(const char* logstr)
{
    int ec;

    // Send an action to all of the worker threads
    ec = pthread_mutex_lock(&done_mutex);
    if (ec)
    {
        clog.error("%s - failed to lock done_mutex in wait_on_workers - %s",
		   logstr, strerror(ec));
    }
    while (done_count < num_threads)
    {
	ec = pthread_cond_wait(&done_cv, &done_mutex);
	if (ec)
	{
	    clog.error("%s - failed wait on done_cv in wait_on_workers - %s",
		       logstr, strerror(ec));
	}

    }
    // Reset count for next time around whilst we have the mutex
    done_count = 0;
    ec = pthread_mutex_unlock(&done_mutex);
    if (ec)
    {
        clog.error("%s - failed to unlock done_mutex in wait_on_workers - %s",
		   logstr, strerror(ec));
    }
}

void
QN_MLP_ThreadFlVar::worker(size_t threadno)
{
    int ec;			// Error code
    enum Action action;		// Our copy of the action we are to do
    unsigned int last_action_seq = 0; // Last action sequence no.
    size_t i;			// Counter
    int exiting = 0;		// Set to true when exiting.
    float nan = qn_nan_f();

    clog.log(QN_LOG_PER_RUN, "Thread %d up and self-aware", threadno);


    // Sized of our part of bunch.
    const size_t w_num_threads = num_threads;
    const size_t w_size_bunch = \
	(size_bunch + w_num_threads - 1)/ w_num_threads;
    const size_t w_first_frame = threadno * w_size_bunch;
    // Our copies of important net params
    size_t w_n_layers = n_layers;
    size_t w_n_weightmats = n_weightmats;
    size_t w_layer_units[MAX_LAYERS];
    size_t w_layer_size[MAX_LAYERS];

    qn_copy_z_vz(MAX_LAYERS, (size_t) -1, w_layer_units);
    qn_copy_z_vz(MAX_LAYERS, (size_t) -1, w_layer_size);
    for (i=0; i<w_n_layers; i++)
    {
	w_layer_units[i] = layer_units[i];
	w_layer_size[i] = w_layer_units[i] * w_size_bunch;
    }
    size_t w_weights_size[MAX_WEIGHTMATS];
    qn_copy_z_vz(MAX_WEIGHTMATS, (size_t) -1, w_weights_size);
    for (i=0; i<w_n_weightmats; i++)
	w_weights_size[i] = weights_size[i];

    // Offsets for reading inputs and writing results
    const size_t w_size_input = w_layer_size[0];
    const size_t w_size_output = w_layer_size[w_n_layers-1];
    const size_t w_input_offset = w_size_input * threadno;
    const size_t w_output_offset = w_size_output * threadno;
    // Other state
    const enum QN_OutputLayerType w_out_layer_type = out_layer_type;
    
    //cz277 - linearity, mul actv
    const enum QN_LayerType *w_hiddenlayer_types = hiddenlayer_types;
    //cz277 - criteria
    const enum QN_CriterionType w_criterion_type = criterion_type;

    // Allocate our local state
    float* w_layer_x[MAX_LAYERS];
    float* w_layer_y[MAX_LAYERS];
    float* w_layer_dedy[MAX_LAYERS];
    float* w_layer_dydx[MAX_LAYERS];
    float* w_layer_dedx[MAX_LAYERS];
    float* w_delta_bias[MAX_LAYERS];
    for (i=0; i<MAX_LAYERS; i++)
    {
	w_layer_x[i] = NULL;
	w_layer_y[i] = NULL;
	w_layer_dedy[i] = NULL;
	w_layer_dydx[i] = NULL;
	w_layer_dedx[i] = NULL;
	w_delta_bias[i] = NULL;
    }
    for (i=1; i<w_n_layers; i++)
    {
	size_t this_layer_space, this_layer_oneframe_space;

	this_layer_space = w_layer_size[i] + CACHE_PAD;
	this_layer_oneframe_space = w_layer_units[i] + CACHE_PAD;

	w_layer_x[i] = new float[this_layer_space];
	qn_copy_f_vf(this_layer_space, nan, w_layer_x[i]);
	w_layer_y[i] = new float[this_layer_space];
	qn_copy_f_vf(this_layer_space, nan, w_layer_y[i]);
	w_layer_dedy[i] = new float[this_layer_space];
	qn_copy_f_vf(this_layer_space, nan, w_layer_dedy[i]);
	w_layer_dydx[i] = new float[this_layer_space];
	qn_copy_f_vf(this_layer_space, nan, w_layer_dydx[i]);
	w_layer_dedx[i] = new float[this_layer_space];
	qn_copy_f_vf(this_layer_space, nan, w_layer_dedx[i]);
	// Biases are one frame not n frames.
	w_delta_bias[i] = new float[this_layer_oneframe_space];
	qn_copy_f_vf(this_layer_oneframe_space, nan, w_delta_bias[i]);
	per_thread[threadno].delta_biases[i] = w_delta_bias[i];
    }

    float* w_delta_weights[MAX_WEIGHTMATS];
    for (i=0; i<MAX_WEIGHTMATS; i++)
    {
	w_delta_weights[i] = NULL;
    }
    for (i=0; i<w_n_weightmats; i++)
    {
	size_t this_weightmat_space = w_weights_size[i]+CACHE_PAD;
	
	w_delta_weights[i] = new float[this_weightmat_space];
	qn_copy_f_vf(this_weightmat_space, nan, w_delta_weights[i]);
	per_thread[threadno].delta_weights[i] = w_delta_weights[i];
    }

    size_t w_cur_layer;		// The index of the current layer.
    size_t w_prev_layer;	// The index of the previous layer.
    size_t w_cur_weinum;	// The index of the current weight matrix.
    size_t w_cur_layer_units;	// The number of units in the current layer.
    size_t w_prev_layer_units;	// The number of units in the previous layer.
    size_t w_cur_layer_size;	// The size of the current layer.
    size_t w_cur_weights_size;	// The size of the current weight matrix.
    float* w_cur_layer_x;	// Input to the current layer non-linearity.
    float* w_cur_layer_y;	// Output from the current layer
				// non-linearity.
    const float* w_prev_layer_y; // Output from the previous non-linearity.
    float* w_cur_layer_dydx;	// dydx for the current layer.
    float* w_cur_layer_dedy;	// dedy for the current layer.
    float* w_prev_layer_dedy;	// dedy for the previous layer.
    float* w_cur_layer_dedx;	// dedx for the current layer.
 
    float* w_cur_layer_bias;	// Biases for the current layer.
    float* w_cur_layer_delta_bias;	// Delta biases for the current layer.
    float* w_cur_weights;	// Weights inputing to the current layer.
    float* w_cur_delta_weights;	// Delta weights inputing to the current layer.

    size_t w_cur_frames;	// Number of frames this time around.
    const float* w_in;		// Input frame pointer
    float* w_out;		// Output frame pointer
    const float* w_target;	// Target frame pointer
    float w_cur_neg_weight_learnrate; // Negative weight learning rate.
    float w_cur_neg_bias_learnrate; // Negative bias learning rate.

    // The main worker loop
    // Exited by a call to pthread_exit()
    while(!exiting)
    {
	//  Wait to be told what to do by sitting on the action
	// ...condition variable waiting for action_seq to change
	ec = pthread_mutex_lock(&action_mutex);
	if (ec)
	    clog.error("failed to lock action_mutex in worker");
	clog.log(QN_LOG_PER_BUNCH, "Thread %d waiting.", threadno); //??
	while (action_seq == last_action_seq)
	{
	    ec = pthread_cond_wait(&action_cv, &action_mutex);
	    if (ec)
		clog.error("failed to wait on action_cv in worker");
	}
	action = action_command;
	last_action_seq = action_seq;
	ec = pthread_mutex_unlock(&action_mutex);
	if (ec)
	    clog.error("failed to unlock action_mutex in worker");

	switch(action)
	{
	case ACTION_EXIT:
	    exiting = 1;
	    break;
	case ACTION_FORWARD:
	    // Work out how many frames we deal with.
	    if (action_n_frames>w_first_frame)
	    {
		w_cur_frames = \
		    qn_min_zz_z(action_n_frames - w_first_frame, w_size_bunch);
	    }
	    else
		w_cur_frames = 0;
	    clog.log(QN_LOG_PER_BUNCH, "Thread %d forward bunch %lu frames.",
		     threadno, w_cur_frames);
	    if (w_cur_frames>0)
	    {
		w_in = action_in + w_input_offset;
		w_out = action_out + w_output_offset;

		// Do all layers except layer 0.
		for (w_cur_layer=1; w_cur_layer<w_n_layers; w_cur_layer++)
		{
		    w_prev_layer = w_cur_layer - 1;
		    w_cur_weinum = w_cur_layer - 1;
		    w_cur_layer_units = w_layer_units[w_cur_layer];
		    w_prev_layer_units = w_layer_units[w_prev_layer];
		    w_cur_layer_size = w_cur_layer_units * w_cur_frames;
		    w_cur_layer_x = w_layer_x[w_cur_layer];
		    w_cur_layer_y = w_layer_y[w_cur_layer];
		    if (w_cur_layer==1)
			w_prev_layer_y = w_in;
		    else
			w_prev_layer_y = w_layer_y[w_prev_layer];
		    w_cur_layer_bias = layer_bias[w_cur_layer];
		    w_cur_weights = weights[w_cur_weinum];

		    qn_copy_vf_mf(w_cur_frames, w_cur_layer_units,
				  w_cur_layer_bias, w_cur_layer_x);
		    qn_mulntacc_mfmf_mf(w_cur_frames, w_prev_layer_units,
					w_cur_layer_units, w_prev_layer_y,
					w_cur_weights, w_cur_layer_x);
		    if (w_cur_layer!=w_n_layers-1)
		    {
			//cz277 - nonlinearity, mul actv
			// This is the intermediate layer non-linearity.
    			switch(w_hiddenlayer_types[w_cur_layer])
    			{
    			case QN_LAYER_LINEAR:
        			qn_copy_vf_vf(w_cur_layer_size, w_cur_layer_x, w_cur_layer_y);
        			break;
    			case QN_LAYER_SIGMOID:
        			qn_sigmoid_vf_vf(w_cur_layer_size, w_cur_layer_x, w_cur_layer_y);
        			break;
    			case QN_LAYER_SOFTMAX:
    			{
        			size_t i;
        			float* w_layer_x_p = w_cur_layer_x;
        			float* w_layer_y_p = w_cur_layer_y;

        			for (i = 0; i < w_cur_frames; ++i)
        			{
            			qn_softmax_vf_vf(w_cur_layer_units, w_layer_x_p, w_layer_y_p);
            			w_layer_x_p += w_cur_layer_units;
            			w_layer_y_p += w_cur_layer_units;
        			}
        			break;
    			}
    			case QN_LAYER_TANH:
        			qn_tanh_vf_vf(w_cur_layer_size, w_cur_layer_x, w_cur_layer_y);
        			break;
    			case QN_LAYER_SOFTSIGN:
        			qn_softsign_vf_vf(w_cur_layer_size, w_cur_layer_x, w_cur_layer_y);
        			break;
    			default:
        			assert(0);
    			}
		    }
		    else
		    {
    			//cz277 - nonlinear //cz277 - criteria
    			switch(w_out_layer_type)
    			{
    			case QN_OUTPUT_LINEAR:
        			qn_copy_vf_vf(w_cur_layer_size, w_cur_layer_x, w_out);
        			break;
    			case QN_OUTPUT_SIGMOID:
        			qn_sigmoid_vf_vf(w_cur_layer_size, w_cur_layer_x, w_out);
        			break;
    			case QN_OUTPUT_SOFTMAX:
    			{
        			size_t i;
        			float* w_layer_x_p = w_cur_layer_x;
        			float* w_layer_y_p = w_out;

        			for (i = 0; i < w_cur_frames; ++i)
        			{
            			qn_softmax_vf_vf(w_cur_layer_units, w_layer_x_p, w_layer_y_p);
            			w_layer_x_p += w_cur_layer_units;
            			w_layer_y_p += w_cur_layer_units;
        			}
        			break;
    			}
    			case QN_OUTPUT_TANH:
        			qn_tanh_vf_vf(w_cur_layer_size, w_cur_layer_x, w_out);
        			break;
    			case QN_OUTPUT_SOFTSIGN:
        			qn_softsign_vf_vf(w_cur_layer_size, w_cur_layer_x, w_out);
        			break;
    			default:
        			assert(0);
    			}
		    }
		}
	    }
	    break;
	case ACTION_TRAIN:
	    // Note that train includes the forward pass.

	    // Work out how many frames we deal with.
	    if (action_n_frames>w_first_frame)
	    {
		w_cur_frames = \
		    qn_min_zz_z(action_n_frames - w_first_frame, w_size_bunch);
	    }
	    else
		w_cur_frames = 0;
	    clog.log(QN_LOG_PER_BUNCH, "Thread %d train bunch %lu frames.",
		     threadno, w_cur_frames);

	    if (w_cur_frames>0)
	    {
		w_in = action_in + w_input_offset;
		w_out = action_out + w_output_offset;
		w_target = action_target + w_output_offset;

		// Do the forward pass.
		// Do all layers except layer 0.
		for (w_cur_layer=1; w_cur_layer<w_n_layers; w_cur_layer++)
		{
		    w_prev_layer = w_cur_layer - 1;
		    w_cur_weinum = w_cur_layer - 1;
		    w_cur_layer_units = w_layer_units[w_cur_layer];
		    w_prev_layer_units = w_layer_units[w_prev_layer];
		    w_cur_layer_size = w_cur_layer_units * w_cur_frames;
		    w_cur_layer_x = w_layer_x[w_cur_layer];
		    w_cur_layer_y = w_layer_y[w_cur_layer];
		    if (w_cur_layer==1)
			w_prev_layer_y = w_in;
		    else
			w_prev_layer_y = w_layer_y[w_prev_layer];
		    w_cur_layer_bias = layer_bias[w_cur_layer];
		    w_cur_weights = weights[w_cur_weinum];

		    qn_copy_vf_mf(w_cur_frames, w_cur_layer_units,
				  w_cur_layer_bias, w_cur_layer_x);
		    qn_mulntacc_mfmf_mf(w_cur_frames, w_prev_layer_units,
					w_cur_layer_units, w_prev_layer_y,
					w_cur_weights, w_cur_layer_x);
		    if (w_cur_layer!=w_n_layers-1)
		    {
			//cz277 - nonlinearity, mul actv
    			switch(w_hiddenlayer_types[w_cur_layer])
    			{
    			case QN_LAYER_LINEAR:
        			qn_copy_vf_vf(w_cur_layer_size, w_cur_layer_x, w_cur_layer_y);
        			break;
    			case QN_LAYER_SIGMOID:
        			qn_sigmoid_vf_vf(w_cur_layer_size, w_cur_layer_x, w_cur_layer_y);
        			break;
    			case QN_LAYER_SOFTMAX:
    			{
        			size_t i;
        			float* w_layer_x_p = w_cur_layer_x;
        			float* w_layer_y_p = w_cur_layer_y;

        			for (i = 0; i < w_cur_frames; ++i)
        			{
            			qn_softmax_vf_vf(w_cur_layer_units, w_layer_x_p, w_layer_y_p);
            			w_layer_x_p += w_cur_layer_units;
            			w_layer_y_p += w_cur_layer_units;
        			}
        			break;
    			}
    			case QN_LAYER_TANH:
        			qn_tanh_vf_vf(w_cur_layer_size, w_cur_layer_x, w_cur_layer_y);
        			
                //cz277 - debug
   		//if (threadno == 0)             
		//for (int k = 0; k < 20; ++k)
                //        printf(" %d: %f ", w_cur_layer, w_cur_layer_y[k]);
                //printf("\n");

				break;
    			case QN_LAYER_SOFTSIGN:
        			qn_softsign_vf_vf(w_cur_layer_size, w_cur_layer_x, w_cur_layer_y);
        			break;
    			default:
        			assert(0);
    			}
		    }
		    else
		    {
			//cz277 - nonlinearity
    			switch(w_out_layer_type)
    			{
    			case QN_OUTPUT_LINEAR:
        			qn_copy_vf_vf(w_cur_layer_size, w_cur_layer_x, w_out);
        			break;
    			case QN_OUTPUT_SIGMOID:
        			qn_sigmoid_vf_vf(w_cur_layer_size, w_cur_layer_x, w_out);
        			break;
    			case QN_OUTPUT_SOFTMAX:
    			{
        			size_t i;
        			float* w_layer_x_p = w_cur_layer_x;
        			float* w_layer_y_p = w_out;

        			for (i = 0; i < w_cur_frames; ++i)
        			{
            			qn_softmax_vf_vf(w_cur_layer_units, w_layer_x_p, w_layer_y_p);
            			w_layer_x_p += w_cur_layer_units;
            			w_layer_y_p += w_cur_layer_units;
        			}
        			break;
    			}
    			case QN_OUTPUT_TANH:
        			qn_tanh_vf_vf(w_cur_layer_size, w_cur_layer_x, w_out);
        			break;
    			case QN_OUTPUT_SOFTSIGN:
        			qn_softsign_vf_vf(w_cur_layer_size, w_cur_layer_x, w_out);
        			break;
    			default:
        			assert(0);
    			}
		    }
		}

		// Iterate back over all layers but the first.
		for (w_cur_layer=w_n_layers-1; w_cur_layer>0; w_cur_layer--)
		{
		    w_prev_layer = w_cur_layer - 1;
		    w_cur_weinum = w_cur_layer - 1;
		    w_cur_layer_units = w_layer_units[w_cur_layer];
		    w_prev_layer_units = w_layer_units[w_prev_layer];
		    w_cur_layer_size = w_cur_layer_units * w_cur_frames;
		    w_cur_weights_size = w_weights_size[w_cur_weinum];
		    w_cur_layer_x = w_layer_x[w_cur_layer];
		    w_cur_layer_y = w_layer_y[w_cur_layer];
		    if (w_cur_layer==1)
			w_prev_layer_y = w_in;
		    else
			w_prev_layer_y = w_layer_y[w_prev_layer];
		    w_cur_layer_dydx = w_layer_dydx[w_cur_layer];
		    w_cur_layer_dedy = w_layer_dedy[w_cur_layer];
		    w_prev_layer_dedy = w_layer_dedy[w_prev_layer];
		    w_cur_layer_dedx = w_layer_dedx[w_cur_layer];
		    w_cur_layer_bias = layer_bias[w_cur_layer];
		    w_cur_layer_delta_bias = w_delta_bias[w_cur_layer];
		    w_cur_weights = weights[w_cur_weinum];
		    w_cur_delta_weights = w_delta_weights[w_cur_weinum];

		    w_cur_neg_weight_learnrate =
			neg_weight_learnrate[w_cur_weinum];
		    w_cur_neg_bias_learnrate = 
			neg_bias_learnrate[w_cur_layer];

	    
		    if ( (w_cur_layer != w_n_layers-1)
			 && backprop_weights[w_cur_weinum+1] )
		    {
			//cz277 - nonlinearity, mul actv
			switch(w_hiddenlayer_types[w_cur_layer])
			{
			case QN_LAYER_LINEAR:
    				qn_copy_vf_vf(w_cur_layer_size, w_cur_layer_dedy, w_cur_layer_dedx);
    				break;
			case QN_LAYER_SIGMOID:
    				qn_dsigmoid_vf_vf(w_cur_layer_size, w_cur_layer_y, w_cur_layer_dydx);
    				qn_mul_vfvf_vf(w_cur_layer_size, w_cur_layer_dydx, w_cur_layer_dedy, w_cur_layer_dedx);
    				break;
			case QN_LAYER_SOFTMAX:
    				qn_dsoftmax_vf_vf(w_cur_layer_size, w_cur_layer_y, w_cur_layer_dydx);
    				qn_mul_vfvf_vf(w_cur_layer_size, w_cur_layer_dydx, w_cur_layer_dedy, w_cur_layer_dedx);
    				break;
			case QN_LAYER_TANH:
    				qn_dtanh_vf_vf(w_cur_layer_size, w_cur_layer_y, w_cur_layer_dydx);
    				qn_mul_vfvf_vf(w_cur_layer_size, w_cur_layer_dydx, w_cur_layer_dedy, w_cur_layer_dedx);
    				                            
				//cz277 - debug
				//if (threadno == 0)
                                //for (int k = 0; k < 20; ++k)
                                //        printf(" %d: %f ", w_cur_layer, w_cur_layer_y[k]);
                                //printf("\n");

				break;
			case QN_LAYER_SOFTSIGN:
    				qn_dsoftsign_vf_vf(w_cur_layer_size, w_cur_layer_y, w_cur_layer_dydx);
    				qn_mul_vfvf_vf(w_cur_layer_size, w_cur_layer_dydx, w_cur_layer_dedy, w_cur_layer_dedx);
    				break;
			default:
    				assert(0);
			}
		    }
		    else
		    {

		        //cz277 - nonlinearity      //cz277 - criteria
		        // Going back through the output layer.
		        switch(w_out_layer_type)
		        {
		        case QN_OUTPUT_LINEAR:
    			    if (w_criterion_type == QN_CRITERION_QUADRATIC) {
           			qn_sub_vfvf_vf(w_cur_layer_size, w_out, w_target, w_cur_layer_dedx);    //dx = dy
    			    } else {        //xentropy
           			qn_dxentropy_vf_vf(w_cur_layer_size, w_out, w_target, w_cur_layer_dedx);
    			    }
    			    break;
		        case QN_OUTPUT_SIGMOID:
    			    if (w_criterion_type == QN_CRITERION_QUADRATIC) {
           			// For a sigmoid layer, de/dx = de/dy . dy/dx
           			qn_sub_vfvf_vf(w_cur_layer_size, w_out, w_target, w_cur_layer_dedy);
           			qn_dsigmoid_vf_vf(w_cur_layer_size, w_out, w_cur_layer_dydx);
           			qn_mul_vfvf_vf(w_cur_layer_size, w_cur_layer_dydx, w_cur_layer_dedy, w_cur_layer_dedx);
    			    } else {        //xentropy
           			qn_sub_vfvf_vf(w_cur_layer_size, w_out, w_target, w_cur_layer_dedx);    //dx = dy
    			    }
    			    break;
		        case QN_OUTPUT_SOFTMAX:
    			    if (w_criterion_type == QN_CRITERION_QUADRATIC) {
           			qn_sub_vfvf_vf(w_cur_layer_size, w_out, w_target, w_cur_layer_dedy);
           			qn_dsoftmax_vf_vf(w_cur_layer_size, w_out, w_cur_layer_dydx);
           			qn_mul_vfvf_vf(w_cur_layer_size, w_cur_layer_dydx, w_cur_layer_dedy, w_cur_layer_dedx);
   			    } else {        //xentropy
           			qn_sub_vfvf_vf(w_cur_layer_size, w_out, w_target, w_cur_layer_dedx);    //dx = dy
   			    }
   			    break;
		        case QN_OUTPUT_TANH:
   			    if (w_criterion_type == QN_CRITERION_QUADRATIC) {
           			// tanh output layer very similar to sigmoid
           			qn_sub_vfvf_vf(w_cur_layer_size, w_out, w_target, w_cur_layer_dedy);
           			qn_dtanh_vf_vf(w_cur_layer_size, w_out, w_cur_layer_dydx);
          			qn_mul_vfvf_vf(w_cur_layer_size, w_cur_layer_dydx, w_cur_layer_dedy, w_cur_layer_dedx);
    			    } else {        //xentropy
            			qn_dxentropy_vf_vf(w_cur_layer_size, w_out, w_target, w_cur_layer_dedy);
            			qn_dtanh_vf_vf(w_cur_layer_size, w_out, w_cur_layer_dydx);
            			qn_mul_vfvf_vf(w_cur_layer_size, w_cur_layer_dydx, w_cur_layer_dedy, w_cur_layer_dedx);
    			    	////qn_sub_vfvf_vf(w_cur_layer_size, w_out, w_cur_layer_dedy, w_cur_layer_dedx);	//wll
			    }
    			    break;
		        case QN_OUTPUT_SOFTSIGN:
    			    if (w_criterion_type == QN_CRITERION_QUADRATIC) {
           			qn_sub_vfvf_vf(w_cur_layer_size, w_out, w_target, w_cur_layer_dedy);
          			qn_dsoftsign_vf_vf(w_cur_layer_size, w_out, w_cur_layer_dydx);
            			qn_mul_vfvf_vf(w_cur_layer_size, w_cur_layer_dydx, w_cur_layer_dedy, w_cur_layer_dedx);
    			    } else {        //xentropy
            			qn_dxentropy_vf_vf(w_cur_layer_size, w_out, w_target, w_cur_layer_dedy);
            			qn_dsoftsign_vf_vf(w_cur_layer_size, w_out, w_cur_layer_dydx);
            			qn_mul_vfvf_vf(w_cur_layer_size, w_cur_layer_dydx, w_cur_layer_dedy, w_cur_layer_dedx);
    			    }
    			    break;
		        default:
    			    assert(0);
		        } // End of output layer type switch.
		    }

		    // Back propogate error through this layer.
		    if (w_cur_layer!=1 && backprop_weights[w_cur_weinum])
		    {
			qn_mul_mfmf_mf(w_cur_frames, w_cur_layer_units,
				       w_prev_layer_units,
				       w_cur_layer_dedx,
				       w_cur_weights, 
				       w_prev_layer_dedy);
		    }
		    // Update weight deltas.
		    if (w_cur_neg_weight_learnrate!=0.0f)
		    {
			qn_copy_f_vf(w_cur_weights_size, 0.0f,
				     w_cur_delta_weights);
			qn_multnacc_fmfmf_mf(w_cur_frames, w_cur_layer_units,
					     w_prev_layer_units,
					     w_cur_neg_weight_learnrate,
					     w_cur_layer_dedx,
					     w_prev_layer_y,
					     w_cur_delta_weights);
		    }
		    // Update bias deltas.
		    if (w_cur_neg_bias_learnrate!=0.0f)
		    {
			qn_sumcol_mf_vf(w_cur_frames, w_cur_layer_units,
					w_cur_layer_dedx,
					w_cur_layer_delta_bias);
		    }
		} // End iteration over layers.
	    } // End of "if frames to do"
	    break;
	default:
	    // Unknown action
	    assert(0);
	}
	
	// Signal that we're done
	ec = pthread_mutex_lock(&done_mutex);
	if (ec)
	    clog.error("failed to lock done_mutex in worker");
	done_count++;			// Indicate one more thread is done.
	if (done_count==num_threads)
	{
	    ec = pthread_cond_signal(&done_cv);
	    if (ec)
		clog.error("failed to signal cond_cv in worker");
	}
	ec = pthread_mutex_unlock(&done_mutex);

    }

    for (i=0; i<w_n_weightmats; i++)
    {
	delete[] w_delta_weights[i];
    }
    for (i=1; i<w_n_layers; i++)
    {
	delete[] w_delta_bias[i];
	delete[] w_layer_dedx[i];
	delete[] w_layer_dydx[i];
	delete[] w_layer_dedy[i];
	delete[] w_layer_y[i];
	delete[] w_layer_x[i];
    }

    clog.log(QN_LOG_PER_RUN, "Thread %d exited.", threadno);
    pthread_exit(NULL);
}

extern "C" {

static void*
worker_wrapper(void *args)
{
    QN_MLP_ThreadFlVar* mlp;
    size_t threadno;

    QN_MLP_ThreadFlVar_WorkerArg* unvoided_args = \
	(QN_MLP_ThreadFlVar_WorkerArg*) args;
    mlp = unvoided_args->mlp;
    threadno = unvoided_args->threadno;

    mlp->worker(threadno);

    return NULL;
}

}; // extern "C"

#endif // #ifde QN_HAVE_LIBPTHREAD

