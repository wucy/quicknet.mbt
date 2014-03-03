const char* QN_MLP_BunchCudaVar_rcsid =
    "$Header: /u/drspeech/repos/quicknet2/QN_MLP_BunchCudaVar.cu,v 1.5 2011/05/24 02:03:14 davidj Exp $";

/* Must include the config.h file first */
#include <QN_config.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "QN_types.h"
#include "QN_Logger.h"
#include "QN_CudaUtils.h"
#include "QN_MLP_BunchCudaVar.h"
#include "QN_fltvec.h"
#include "QN_intvec.h"
#include "QN_cuvec.h"

#include <cuda.h>
#include <cublas.h>


// These appear later but we do not want them in the header file
// __global__ void QN_BunchCudaVar_forward_bunch(QN_BunchCudaVar_Workspace *ws,
// 					      int n_frames);
// __global__ void QN_BunchCudaVar_train_bunch(QN_BunchCudaVar_Workspace *ws,
// 					    int n_frames);
// __device__ void QN_BunchCudaVar_forward_bunch_do(QN_BunchCudaVar_Workspace *ws,
//					      int n_frames);

float * QN_MLP_BunchCudaVar::Last2_weights()
{
    fromdev_vf_vf("weights", 1000 * 2698, dev.weights[n_layers - 1], dev.cache_last2_weights); //cw564 - mbt -- TODO
    return dev.cache_last2_weights;
}

float * QN_MLP_BunchCudaVar::Last_y()
{
    return dev.cache_raw_last2_y;
}

QN_MLP_BunchCudaVar::QN_MLP_BunchCudaVar(int a_bp_num_layer,	//cz277 - nn fea bp
					 int a_debug,
					 const char* a_dbgname,
					 size_t a_n_layers,
					 const size_t a_layer_units[QN_MLP_MAX_LAYERS],
					 enum QN_CriterionType a_criteriontype,     //cz277 - criteria
					 enum QN_LayerType *a_layertype,     //cz277 - nonlinearity, mul actv
					 enum QN_OutputLayerType a_outtype,
					 size_t a_size_bunch,
					 int device_no,	//cz277 - device select
					 const char *env_var4dev_id)	//cz277 - env var
    : QN_MLP_BaseFl(a_debug, a_dbgname, "QN_MLP_BunchCudaVar",
		    a_size_bunch, a_n_layers,
		    a_layer_units),	//cz277 - dnn
      bp_num_layer(a_bp_num_layer),	//cz277 - nn fea bp
      criterion_type(a_criteriontype),  //cz277 - criteria
      hiddenlayer_types(a_layertype),    //cz277 - nonlinearity, mul actv
      out_layer_type(a_outtype)
{
    size_t i;

    // Initialize CUDA if it has not happened already

    QN_cuda_init(device_no, env_var4dev_id);	//cz277 - device select, env var
    
    // Some stuff so that when things go wrong it is more obvious.
    // for (i=0; i<MAX_LAYERS; i++)
    // {
    // 	layer_x[i] = NULL;
    // 	layer_y[i] = NULL;
    // 	layer_dedy[i] = NULL;
    // 	layer_dydx[i] = NULL;
    // 	layer_dedx[i] = NULL;
    // 	layer_delta_bias[i] = NULL;
    // }
    
    //cz277 - criteria
    switch(criterion_type)
    {
    case QN_CRITERION_QUADRATIC:
    case QN_CRITERION_XENTROPY:
            break;
    default:
            clog.error("Failed to create an MLP with an invalid training criterion type.");
    }

    //cz277 - debug
    /*for (int i = 1; i < a_n_layers; ++i)
        printf("layer %d, srcaddr = %x, srcval = %d, dstaddr = %x, dstval = %d\n", i, a_layertype, a_layertype[2], hiddenlayer_types, hiddenlayer_types[2]);*/

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
            clog.error("Failed to create an MLP with an invalid hidden layer out type.");
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
	     clog.error("Failed to create an MLP with an invalid output layer type.");
    }
    if (size_bunch == 0)
	clog.error("Cannot use a 0 bunch size.");


    // Allocate device data structures

    size_t in_size = layer_size[0];
    size_t out_size = layer_size[n_layers-1];

    devnew_vf("in", in_size, &(dev.in));
    devnew_vf("out", out_size, &(dev.out));
    devnew_vf("targ", out_size, &(dev.targ));

    //cz277 - fast softmax
    
    devnew_vf("compcache", out_size, &(dev.compcache));

    for (i = 1; i<n_layers; i++)
    {
	size_t size = layer_size[i];
	size_t units = layer_units[i];

	devnew_vf("layer_bias", size, &(dev.layer_bias[i]));
	devnew_vf("layer_y", size, &(dev.layer_y[i]));
	devnew_vf("layer_x", size, &(dev.layer_x[i]));
	devnew_vf("layer_dedy", size, &(dev.layer_dedy[i]));
	devnew_vf("layer_dydx", size, &(dev.layer_dydx[i]));
	devnew_vf("layer_dedx", size, &(dev.layer_dedx[i]));
	devnew_vf("layer_delta_bias", units, &(dev.layer_delta_bias[i]));

	//cz277 - revisit momentum
	devnew_vf("l_bias_delta", size, &(dev.l_bias_delta[i]));
    }
    // Set up the per-weight-matrix data structures.
    for (i = 0; i<n_weightmats; i++)
    {
	// Note the host weights are alloacted by QN_MLP_BaseFl 
	size_t n_weights = weights_size[i];

	// Allocate device data structures
	devnew_vf("weights", n_weights, &dev.weights[i]);
    
	//cz277 - momentum
        devnew_vf("weights_delta", n_weights, &dev.weights_delta[i]);
    }

    clog.log(QN_LOG_PER_RUN, "Created net with %lu layers, bunchsize %lu.",
	     n_layers, size_bunch);
    for (i=0; i<n_layers; i++)
    {
	clog.log(QN_LOG_PER_RUN, "Layer %lu has %lu units.",
		 i+1, layer_units[i]);
    }
    dev_weights_stale = QN_TRUE;
    host_weights_stale = QN_FALSE;
    
    dev.cache_raw_last2_y = new float[300000000]; //cw564 - mbt -- TODO
    dev.cache_weights = new float[300000000]; //cw564 - mbt -- TODO
    dev.cache_last2_weights = new float[300000000];
}

QN_MLP_BunchCudaVar::~QN_MLP_BunchCudaVar()
{
    size_t i;

    QN_cuda_check();
    // Wind down the per-weight-matrix data structures.
    for (i = 0; i<n_weightmats; i++)
    {
	// Deallocate device data structures
	devfree_vf("weights", dev.weights[i]);

        //cz277 - momentum
	devfree_vf("weights_delta", dev.weights_delta[i]);

	// Note the host weights are deallocated by QN_MLP_BaseFl 
    }
    // Wind down the per-layer data structures.
    for (i = 1; i<n_layers; i++)
    {
	// delete [] layer_y[i];
	// delete [] layer_delta_bias[i];
	// delete [] layer_dedx[i];
	// delete [] layer_dydx[i];
	// delete [] layer_dedy[i];
	// delete [] layer_x[i];
	// Note the host biases are deallocated by QN_MLP_BaseFl 

	devfree_vf("layer_delta_bias", dev.layer_delta_bias[i]);
	devfree_vf("layer_dedx", dev.layer_dedx[i]);
	devfree_vf("layer_dydx", dev.layer_dydx[i]);
	devfree_vf("layer_dedy", dev.layer_dedy[i]);
	devfree_vf("layer_x", dev.layer_x[i]);
	devfree_vf("layer_y", dev.layer_y[i]);
	devfree_vf("layer_bias", dev.layer_bias[i]);

	//cz277 - revisit momentum
	devfree_vf("l_bias_delta", dev.l_bias_delta[i]);
    }
    devfree_vf("targ", dev.targ);
    devfree_vf("out", dev.out);
    devfree_vf("in", dev.in);
    
    //cz277 - fast softmax
    devfree_vf("compcache", dev.compcache);

    //cw564 - mbt
    delete [] dev.cache_raw_last2_y;
    delete [] dev.cache_weights;
}



void
QN_MLP_BunchCudaVar::forward_bunch(size_t n_frames, const float* in, float* out, const float * * spkr_wgt, const size_t num_basis)
{

//    printf("in=%x, out=%x\n", in, out);

    // Copy the data across to the device
    int in_size = n_frames * layer_units[0];
    int out_size = n_frames * layer_units[n_layers-1];
    todev_vf_vf("forward_bunch().in", in_size, in, dev.in);


    size_t cur_layer;		// The index of the current layer.
    size_t prev_layer;		// The index of the previous layer.
    size_t cur_weinum;		// The index of the current weight matrix.
    size_t cur_layer_units;	// The number of units in the current layer.
    size_t prev_layer_units;	// The number of units in the previous layer.
    size_t cur_layer_size;	// The size of the current layer.
    float* cur_layer_x;		// Input to the current layer non-linearity.
    float* cur_layer_y;		// Output from the current layer
				// non-linearity.
    float* prev_layer_y;	// Output from the previous non-linearity.
    float* cur_layer_bias;	// Biases for the current layer.
    float* cur_weights;		// Weights inputing to the current layer.

    // Iterate over all of the layers except the input.  This is just one 
    // iteration for 2-layer MLPs.
    // Note that layer index starts at 0 for inputlayer, so we start at 1.
    for (cur_layer=1; cur_layer<n_layers; cur_layer++)
    {
        prev_layer = cur_layer - 1;
        cur_weinum = cur_layer - 1;
        cur_layer_units = layer_units[cur_layer];
        prev_layer_units = layer_units[prev_layer];
        cur_layer_size = cur_layer_units * n_frames;
        cur_layer_x = dev.layer_x[cur_layer];
        cur_layer_y = dev.layer_y[cur_layer];
        if (cur_layer==1)
            prev_layer_y = dev.in;
        else if (cur_layer == n_layers - 1) //cw564 - mbt
        {
            float * h_cache_prev_layer_y = dev.cache_raw_last2_y;
            float * d_prev_layer_y = dev.layer_y[prev_layer];
            fromdev_vf_vf("mbt.ori_prev_layer_y", prev_layer_units * n_frames, 
                    d_prev_layer_y, h_cache_prev_layer_y);


            int old_prev_layer_units = prev_layer_units;
            prev_layer_units /= num_basis;

            float * h_wsum_prev_layer_y = new float[prev_layer_units * n_frames];
            //TODO fast summation
            for (int ff = 0; ff < n_frames; ++ ff)
            {
                for (int now_dim = 0; now_dim < prev_layer_units; ++ now_dim)
                {
                    int new_id = ff * prev_layer_units + now_dim;
                    h_wsum_prev_layer_y[new_id] = 0;
                    for (int bb = 0; bb < num_basis; ++ bb)
                    {
                        int old_id = ff * old_prev_layer_units + bb * prev_layer_units + now_dim;
                        h_wsum_prev_layer_y[new_id] += h_cache_prev_layer_y[old_id] * spkr_wgt[ff][bb];
                    }
                }
            }
            todev_vf_vf("mbt.new_prev_layer_y", prev_layer_units * n_frames, 
                    h_wsum_prev_layer_y, d_prev_layer_y);
            delete [] h_wsum_prev_layer_y;

            prev_layer_y = d_prev_layer_y;
        }
        else
            prev_layer_y = dev.layer_y[prev_layer];
        cur_layer_bias = dev.layer_bias[cur_layer];
        cur_weights = dev.weights[cur_weinum];

        if (checking)
            devcheck("forward_bunch #1");
        //cz277 - fast softmax
        qn_dev_fastcopy_vf_mf(n_frames, cur_layer_units, cur_layer_bias, cur_layer_x);
        //qn_dev_copy_vf_mf(n_frames, cur_layer_units, cur_layer_bias,
        //		    cur_layer_x);
        if (checking)
            devcheck("forward_bunch #2");
        qn_dev_mulntacc_mfmf_mf(n_frames, prev_layer_units, cur_layer_units,
                prev_layer_y, cur_weights,
                cur_layer_x); 


	if (checking)
	    devcheck("forward_bunch #3");
	
	// Check if we are doing things differently for the final layer.
	if (cur_layer!=n_layers - 1)
	{
	    //cz277 - nonlinearity, mul actv
            switch(hiddenlayer_types[cur_layer])
            {
            case QN_LAYER_LINEAR:
            	qn_dev_copy_vf_vf(cur_layer_size, cur_layer_x, cur_layer_y);
            	break;
            case QN_LAYER_SIGMOID:
            	qn_dev_sigmoid_vf_vf(cur_layer_size, cur_layer_x, cur_layer_y);
            	break;
            case QN_LAYER_SOFTMAX:
        	qn_dev_multisoftmax_mf_mf(n_frames, cur_layer_units, cur_layer_x, cur_layer_y);
        	break;
            case QN_LAYER_TANH:
            	qn_dev_tanh_vf_vf(cur_layer_size, cur_layer_x, cur_layer_y);
                break;
            case QN_LAYER_SOFTSIGN:
            	qn_dev_softsign_vf_vf(cur_layer_size, cur_layer_x, cur_layer_y);
            	break;
      	    default:
		//cz277 - debug
		/*printf("curlayer = %d, curtype = %d, linear = %d, sigmoid = %d, softmax = %d, tanh = %d, softsign = %d\n", cur_layer, hiddenlayer_types[cur_layer], QN_LAYER_LINEAR, QN_LAYER_SIGMOID, QN_LAYER_SOFTMAX, QN_LAYER_TANH, QN_LAYER_SOFTSIGN);*/
            	assert(0);
            }
	
	}
	else
	{
            //cz277 - nonlinear //cz277 - criteria
            switch(out_layer_type)
            {
            case QN_OUTPUT_LINEAR:
                qn_dev_copy_vf_vf(cur_layer_size, cur_layer_x, dev.out);
                break;
            case QN_OUTPUT_SIGMOID:
                qn_dev_sigmoid_vf_vf(cur_layer_size, cur_layer_x, dev.out);
                break;
            case QN_OUTPUT_SOFTMAX:
                //qn_dev_multisoftmax_mf_mf(n_frames, cur_layer_units, cur_layer_x, dev.out);
		//cz277 - fast softmax
		qn_dev_fastsoftmax_mf_mf(n_frames, cur_layer_units, cur_layer_x, dev.compcache, dev.out);
		/*fromdev_vf_vf("forward_bunch().out", out_size, cur_layer_x, out);
		printf("outsize = %d, in = %e, ", out_size, out[0]);
		fromdev_vf_vf("forward_bunch().out", out_size, dev.compcache, out);
		fromdev_vf_vf("forward_bunch().out", out_size, dev.out, out);
		printf("out = %e\n", out[0]);*/
                break;
            case QN_OUTPUT_TANH:
                qn_dev_tanh_vf_vf(cur_layer_size, cur_layer_x, dev.out);
                break;
            case QN_OUTPUT_SOFTSIGN:
                qn_dev_softsign_vf_vf(cur_layer_size, cur_layer_x, dev.out);
                break;
            default:
                assert(0);
            }

	}
    }
    // Copy the data back from the device
    fromdev_vf_vf("forward_bunch().out", out_size, dev.out, out);
    if (checking)
	devcheck("forward_bunch #9");

}

void
QN_MLP_BunchCudaVar::train_bunch(size_t n_frames, const float *in,
				 const float* target, float* out, const float * * spkr_wgt, const size_t num_basis)
{
// First move forward, which copies over in and out
    forward_bunch(n_frames, in, out, spkr_wgt, num_basis);
    if (checking)
	devcheck("train_bunch #0");

// So we stil have to copy across targ
    int out_size = n_frames * layer_units[n_layers-1];
    todev_vf_vf("train_bunch().targ", out_size, target, dev.targ);
    if (checking)
	devcheck("train_bunch #1");

    size_t cur_layer;		// The index of the current layer.
    size_t prev_layer;		// The index of the previous layer.
    size_t cur_weinum;		// The index of the current weight matrix.
    size_t cur_layer_units;	// The number of units in the current layer.
    size_t prev_layer_units;	// The number of units in the previous layer.
    size_t cur_layer_size;	// The size of the current layer.
    float* cur_layer_y;		// Output from the current layer
				// non-linearity.
    const float* prev_layer_y;	// Output from the previous non-linearity.
    float* cur_layer_dydx;	// dydx for the current layer.
    float* cur_layer_dedy;	// dedy for the current layer.
    float* prev_layer_dedy;	// dedy for the previous layer.
    float* cur_layer_dedx;	// dedx for the current layer.
    float* cur_layer_bias;	// Biases for the current layer.
    float* cur_layer_delta_bias; // Delta biases for the current layer.
    float* cur_weights;		// Weights inputing to the current layer.

    //cz277 - momentum
    float *cur_weights_delta;

    //cz277 - revisit momentum
    float *cur_l_bias_delta;

    //cz277 - nn fea bp
    int bp_num_layer_idx = this->bp_num_layer;

    //cw564 - mbt
    float * z_spkr = new float[n_frames];
    for (int ff = 0; ff < n_frames; ++ ff)
    {
        z_spkr[ff] = 0;
	for (int bb = 0; bb < num_basis; ++ bb) z_spkr[ff] += spkr_wgt[ff][bb];
    }

    // Iterate back over all layers but the first.
    for (cur_layer=n_layers-1; cur_layer>0 && (bp_num_layer_idx != 0); cur_layer--, --bp_num_layer_idx)	//cz277 - nn fea bp
    {

	prev_layer = cur_layer - 1;
	cur_weinum = cur_layer - 1;
	cur_layer_units = layer_units[cur_layer];
	prev_layer_units = layer_units[prev_layer];
	cur_layer_size = cur_layer_units * n_frames;

	//cw564 - mbt
	if (cur_layer == n_layers - 2)
	{
	    todev_vf_vf("mbt.raw_cur_layer_last2_y", cur_layer_size, 
                    dev.cache_raw_last2_y, dev.layer_y[cur_layer]);
	}
	
	cur_layer_y = dev.layer_y[cur_layer];
	if (cur_layer==1)
	    prev_layer_y = dev.in;
	else
	    prev_layer_y = dev.layer_y[prev_layer];
	cur_layer_dydx = dev.layer_dydx[cur_layer];
	cur_layer_dedy = dev.layer_dedy[cur_layer];
	prev_layer_dedy = dev.layer_dedy[prev_layer];
	cur_layer_dedx = dev.layer_dedx[cur_layer];
	cur_layer_bias = dev.layer_bias[cur_layer];
	cur_layer_delta_bias = dev.layer_delta_bias[cur_layer];
	cur_weights = dev.weights[cur_weinum];
	
	//cz277 - momentum
	cur_weights_delta = dev.weights_delta[cur_weinum];
	
	//cz277 - revisit momentum
	cur_l_bias_delta = dev.l_bias_delta[cur_layer];

	float cur_neg_weight_learnrate = neg_weight_learnrate[cur_weinum];
	float cur_neg_bias_learnrate = neg_bias_learnrate[cur_layer];

	if (cur_layer!=n_layers - 1 && backprop_weights[cur_weinum+1])
	{
            //cz277 - nonlinearity, mul actv
            switch(hiddenlayer_types[cur_layer])
            {
            case QN_LAYER_LINEAR:
                qn_dev_copy_vf_vf(cur_layer_size, cur_layer_dedy, cur_layer_dedx);
                break;
            case QN_LAYER_SIGMOID:
                qn_dev_dsigmoid_vf_vf(cur_layer_size, cur_layer_y, cur_layer_dydx);
                qn_dev_mul_vfvf_vf(cur_layer_size, cur_layer_dydx, cur_layer_dedy, cur_layer_dedx);
                break;
            case QN_LAYER_SOFTMAX:
                qn_dev_dsoftmax_vf_vf(cur_layer_size, cur_layer_y, cur_layer_dydx);
                qn_dev_mul_vfvf_vf(cur_layer_size, cur_layer_dydx, cur_layer_dedy, cur_layer_dedx);
                break;
            case QN_LAYER_TANH:
                qn_dev_dtanh_vf_vf(cur_layer_size, cur_layer_y, cur_layer_dydx);
                qn_dev_mul_vfvf_vf(cur_layer_size, cur_layer_dydx, cur_layer_dedy, cur_layer_dedx);
		break;
            case QN_LAYER_SOFTSIGN:
                qn_dev_dsoftsign_vf_vf(cur_layer_size, cur_layer_y, cur_layer_dydx);
                qn_dev_mul_vfvf_vf(cur_layer_size, cur_layer_dydx, cur_layer_dedy, cur_layer_dedx);
                break;
            default:
                assert(0);
            }

	}
	else
	{
            //cz277 - nonlinearity      //cz277 - criteria
            // Going back through the output layer.
            switch(out_layer_type)
            {
            case QN_OUTPUT_LINEAR:
                if (criterion_type == QN_CRITERION_QUADRATIC) {
                    qn_dev_sub_vfvf_vf(cur_layer_size, dev.out, dev.targ, cur_layer_dedx);    //dx = dy
                } else {        //xentropy
                    qn_dev_dxentropy_vf_vf(cur_layer_size, dev.out, dev.targ, cur_layer_dedx);
                }
                break;
            case QN_OUTPUT_SIGMOID:
                if (criterion_type == QN_CRITERION_QUADRATIC) {
                    // For a sigmoid layer, de/dx = de/dy . dy/dx
                    qn_dev_sub_vfvf_vf(cur_layer_size, dev.out, dev.targ, cur_layer_dedy);
                    qn_dev_dsigmoid_vf_vf(cur_layer_size, dev.out, cur_layer_dydx);
                    qn_dev_mul_vfvf_vf(cur_layer_size, cur_layer_dydx, cur_layer_dedy, cur_layer_dedx);
                } else {        //xentropy
                    qn_dev_sub_vfvf_vf(cur_layer_size, dev.out, dev.targ, cur_layer_dedx);    //dx = dy
                }
                break;
            case QN_OUTPUT_SOFTMAX:
                if (criterion_type == QN_CRITERION_QUADRATIC) {
                    qn_dev_sub_vfvf_vf(cur_layer_size, dev.out, dev.targ, cur_layer_dedy);
                    qn_dev_dsoftmax_vf_vf(cur_layer_size, dev.out, cur_layer_dydx);
                    qn_dev_mul_vfvf_vf(cur_layer_size, cur_layer_dydx, cur_layer_dedy, cur_layer_dedx);
                } else {        //xentropy
		    qn_dev_sub_vfvf_vf(cur_layer_size, dev.out, dev.targ, cur_layer_dedx);    //dx = dy
		    //qn_dev_dxentropy_vf_vf(cur_layer_size, dev.out, dev.targ, cur_layer_dedy);
		    //qn_dev_dsoftmax_vf_vf(cur_layer_size, dev.out, cur_layer_dydx);
		    //qn_dev_mul_vfvf_vf(cur_layer_size, cur_layer_dydx, cur_layer_dedy, cur_layer_dedx);
                }
                break;
            case QN_OUTPUT_TANH:
                if (criterion_type == QN_CRITERION_QUADRATIC) {
                    // tanh output layer very similar to sigmoid
                    qn_dev_sub_vfvf_vf(cur_layer_size, dev.out, dev.targ, cur_layer_dedy);
                    qn_dev_dtanh_vf_vf(cur_layer_size, dev.out, cur_layer_dydx);
                    qn_dev_mul_vfvf_vf(cur_layer_size, cur_layer_dydx, cur_layer_dedy, cur_layer_dedx);
                } else {        //xentropy
                    qn_dev_dxentropy_vf_vf(cur_layer_size, dev.out, dev.targ, cur_layer_dedy);
                    qn_dev_dtanh_vf_vf(cur_layer_size, dev.out, cur_layer_dydx);
                    qn_dev_mul_vfvf_vf(cur_layer_size, cur_layer_dydx, cur_layer_dedy, cur_layer_dedx);
                    //qn_dev_sub_vfvf_vf(cur_layer_size, dev.out, cur_layer_dedy, cur_layer_dedx);	//wll
		}
                break;
            case QN_OUTPUT_SOFTSIGN:
                if (criterion_type == QN_CRITERION_QUADRATIC) {
                    qn_dev_sub_vfvf_vf(cur_layer_size, dev.out, dev.targ, cur_layer_dedy);
                    qn_dev_dsoftsign_vf_vf(cur_layer_size, dev.out, cur_layer_dydx);
                    qn_dev_mul_vfvf_vf(cur_layer_size, cur_layer_dydx, cur_layer_dedy, cur_layer_dedx);
                } else {        //xentropy
                    qn_dev_dxentropy_vf_vf(cur_layer_size, dev.out, dev.targ, cur_layer_dedy);
                    qn_dev_dsoftsign_vf_vf(cur_layer_size, dev.out, cur_layer_dydx);
                    qn_dev_mul_vfvf_vf(cur_layer_size, cur_layer_dydx, cur_layer_dedy, cur_layer_dedx);
                }
                break;
            default:
                assert(0);
            } // End of output layer type switch.

	} // End of special output layer treatment.

	// Back propogate error through this layer.
	if (cur_layer!=1 && backprop_weights[cur_weinum])
	{
	
	//cw564 - mbt
	if (cur_layer == n_layers - 1)
		prev_layer_units = prev_layer_units / num_basis;

	qn_dev_mul_mfmf_mf(n_frames, cur_layer_units, prev_layer_units,
			   cur_layer_dedx, cur_weights, prev_layer_dedy);


	
	//cw564 - mbt -- prev_layer_dedy
        if (cur_layer == n_layers - 1)
	{
	    int size = n_frames * prev_layer_units;
	    float * h_raw_prev_layer_dedy = new float[size];
	    float * h_new_prev_layer_dedy = new float[size * num_basis];

	    fromdev_vf_vf("raw_prev_layer_dedy", size, prev_layer_dedy, h_raw_prev_layer_dedy);

	    int cnt = 0;
	    for (int ff = 0; ff < n_frames; ++ ff)
	    {
	        for (int bb = 0; bb < num_basis; ++ bb)
		{
		    for (int dd = 0; dd < prev_layer_units; ++ dd)
	            {
		        h_new_prev_layer_dedy[cnt] = 
			    h_raw_prev_layer_dedy[ff * prev_layer_units + dd] * spkr_wgt[ff][bb] / z_spkr[ff];
			cnt ++;
                    }
		}
	    }

	    int new_size = size * num_basis;
	    todev_vf_vf("new_prev_layer_dedy", new_size, h_new_prev_layer_dedy, prev_layer_dedy);
	    delete [] h_raw_prev_layer_dedy;
	    delete [] h_new_prev_layer_dedy;
	}

	if (checking)
	    devcheck("train_bunch #12");
	
	}
	// Update weights.
	if (cur_neg_weight_learnrate!=0.0f)
	{


	    //cz277 - momentum
	    qn_dev_multnacc2_fmfmf_mf(n_frames, cur_layer_units, prev_layer_units,
			    cur_neg_weight_learnrate, alpha_momentum, cur_layer_dedx,
			    prev_layer_y, cur_weights_delta);
	    //weights_delta[tau] = -eta * partial_E_div_partial_weights + alpha * weights_delta[tau - 1]

	    //cz277 - weight decay	
	    qn_dev_mulacc_vff_vf(cur_layer_units * prev_layer_units, cur_weights, cur_neg_weight_learnrate * weight_decay_factor, cur_weights_delta);	//weights_delta[tau] = -yita * nu * weights[tau] + weights_delta[tau]


    	    
	    qn_dev_mulacc_vff_vf(
                    cur_layer_units * prev_layer_units, 
                    cur_weights_delta, 1.0, 
                    cur_weights);	//weights[tau + 1] = weights[tau] + weights_delta[tau]		
	    	    
	    //cw564 - mbt -- TODO BEGIN set zero to entries not in diag-blocks
            if (cur_layer <= n_layers - 2)
            {
	        int weight_size = cur_layer_units * prev_layer_units;
	        float * h_cur_weights = dev.cache_weights;
	        fromdev_vf_vf("weights", weight_size, cur_weights, h_cur_weights);
	        for (int i = 0; i < weight_size; ++ i)
                {
                    int col = i % prev_layer_units;
                    int row = i / prev_layer_units;
                    int one_base_col_size = prev_layer_units / num_basis;
                    int one_base_row_size = cur_layer_units / num_basis;
                    int k = min(col / one_base_col_size, row / one_base_row_size);
                    int base_col = col - k * one_base_col_size;
                    int base_row = row - k * one_base_row_size;
		    if (base_col >= one_base_col_size || base_row >= one_base_row_size)
		    {
		        h_cur_weights[i] = 0;
		    }
	        }
	        todev_vf_vf("new_weights", weight_size, h_cur_weights, cur_weights);
            }
	    //cw564 - mbt -- TODO END set zero to entries not in diag-blocks

		
    
	    if (checking)
		devcheck("train_bunch #13");
	}
	// Update biases.
	if (cur_neg_bias_learnrate!=0.0f)
	{
	    qn_dev_sumcol_mf_vf(n_frames, cur_layer_units, cur_layer_dedx,
				cur_layer_delta_bias); 

	    //cz277 - revisit momentum 
	    qn_dev_scale_fvf_vf(cur_layer_units, alpha_momentum, cur_l_bias_delta);	//acquire alpha * bias_delta[tau - 1]
	    qn_dev_mulacc_vff_vf(cur_layer_units, cur_layer_delta_bias, cur_neg_bias_learnrate, cur_l_bias_delta);	//bias_delta[tau] = alpha * bias_delta[tau - 1] + neg_eta * partial_E_div_partial_bias

	    //cz277 - weight decay
	    qn_dev_mulacc_vff_vf(cur_layer_units, cur_layer_bias, cur_neg_weight_learnrate * weight_decay_factor, cur_l_bias_delta);	//bias_delta[tau] = -yita * nu * bias[tau] + bias_delta[tau]

	    qn_dev_mulacc_vff_vf(cur_layer_units, cur_l_bias_delta, 1.0, cur_layer_bias);	//bias[tau + 1] = bias[tau] + bias_delta[tau]

	    if (checking)
		devcheck("train_bunch #15");
	}
    } // End of iteration over all layers.


    // Copy the data back from the device
    fromdev_vf_vf("train_bunch().out", out_size, dev.out, out);
    if (checking)
	devcheck("train_bunch #16");

}

void
QN_MLP_BunchCudaVar::forward(size_t n_frames, const float* in, float* out, const float * * wgt, const size_t num_basis)
{
    refresh_dev_weights();
    QN_MLP_BaseFl::forward(n_frames, in, out, wgt, num_basis);
}

void
QN_MLP_BunchCudaVar::train(size_t n_frames, const float* in,
			   const float* target, float* out, const float * * wgt, const size_t num_basis)
{
    refresh_dev_weights();
    QN_MLP_BaseFl::train(n_frames, in, target, out, wgt, num_basis);
    host_weights_stale = QN_TRUE;
}

void
QN_MLP_BunchCudaVar::set_weights(enum QN_SectionSelector which,
				 size_t row, size_t col,
				 size_t n_rows, size_t n_cols,
				 const float* weights)
{
    refresh_host_weights();
    QN_MLP_BaseFl::set_weights(which, row, col, n_rows, n_cols, weights);
    dev_weights_stale = QN_TRUE;
}


void
QN_MLP_BunchCudaVar::get_weights(enum QN_SectionSelector which,
				 size_t row, size_t col,
				 size_t n_rows, size_t n_cols,
				 float* weights)
{
    refresh_host_weights();
    QN_MLP_BaseFl::get_weights(which, row, col, n_rows, n_cols, weights);
}

void
QN_MLP_BunchCudaVar::refresh_dev_weights(void)
{
    if (dev_weights_stale)
    {
	dev_weights_stale = QN_FALSE;

	size_t i;

	for (i = 0; i<n_weightmats; i++)
	{
	    size_t n_weights;

	    n_weights = weights_size[i]; 
	    todev_vf_vf("refresh_dev_weights().weights",
			n_weights, weights[i], dev.weights[i]);
	    
	    //cz277 - momentum
	    todev_vf_vf("refresh_dev_weights().weights_delta",
			n_weights, weights_delta[i], dev.weights_delta[i]);
	}

	for (i = 1; i<n_layers; i++)
	{
	    size_t n_biases;

	    n_biases = layer_size[i];
	    todev_vf_vf("refresh_dev_weights().layer_bias",
			n_biases, layer_bias[i], dev.layer_bias[i]);

	    //cz277 - revisit momentum
	    todev_vf_vf("refresh_dev_weights().l_bias_delta",
                        n_biases, bias_delta[i], dev.l_bias_delta[i]);
	}
    }
}

void
QN_MLP_BunchCudaVar::refresh_host_weights(void)
{
    if (host_weights_stale)
    {
	host_weights_stale = QN_FALSE;

	size_t i;

	for (i = 0; i<n_weightmats; i++)
	{
	    size_t n_weights;

	    n_weights = weights_size[i]; 
	    fromdev_vf_vf("refresh_host_weights.weights)",
			  n_weights, dev.weights[i], weights[i]);

	    //cz277 - momentum
	    fromdev_vf_vf("refresh_host_weights.weights_delta)",
			  n_weights, dev.weights_delta[i], weights_delta[i]);
	}

	for (i = 1; i<n_layers; i++)
	{
	    size_t n_biases;

	    n_biases = layer_size[i];
	    fromdev_vf_vf("freresh_host_weights().layer_bias", 
			   n_biases, dev.layer_bias[i], layer_bias[i]);

	    //cz277 - revisit momentum
	    fromdev_vf_vf("freresh_host_weights().bias_delta",
                           n_biases, dev.l_bias_delta[i], bias_delta[i]);
	}
    }
}

void
QN_MLP_BunchCudaVar::devnew_vf(const char* varname, int n, float **devptr)
{
    cublasStatus e;

    e = cublasAlloc(n, sizeof(float), (void **) devptr);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
	clog.error("cuda blas device new_vf error variable %s - %s.",
		   varname, QN_cublas_error_string(e));
    }
    clog.log(QN_LOG_PER_EPOCH, "Created CUDA float vec \"%s\" size %i at %.8x\n", varname, n, (unsigned long) *devptr);
}

void
QN_MLP_BunchCudaVar::devnew_vi(const char* varname, int n, int **devptr)
{
    cublasStatus e;

    e = cublasAlloc(n, sizeof(int), (void **) devptr);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
	clog.error("cuda blas device new_vi error variable %s - %s.",
		   varname, QN_cublas_error_string(e));
    }
    clog.log(QN_LOG_PER_EPOCH, "Created CUDA int vec \"%s\" size %i at %.8x\n", varname, n, (unsigned long) *devptr);

}


void 
QN_MLP_BunchCudaVar::devcheck(const char* location)
{
    cudaError_t e;

    e = cudaThreadSynchronize();
    if (e!=cudaSuccess)
    {
	clog.error("asynchronous CUDA error at %s - %s.",
		   location, cudaGetErrorString(e));
    }
    
    cublasStatus eb;

    eb = cublasGetError();
    if (eb!=CUBLAS_STATUS_SUCCESS)
	QN_ERROR("QN_cuda_check", "accumulated cublas error detected");
}

void
QN_MLP_BunchCudaVar::devnew(const char* varname, int n, int size,
			    void **devptr)
{
    cublasStatus e;

    e = cublasAlloc(n, size, devptr);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
	clog.error("cuda blasw device free error variable %s - %s.",
		   varname, QN_cublas_error_string(e));
    }

}

void
QN_MLP_BunchCudaVar::devfree(const char* varname, const void* devptr)
{
    cublasStatus e;
    e = cublasFree((void *)devptr);	//cz277 - cuda
    if (e != CUBLAS_STATUS_SUCCESS)
    {
	clog.error("cuda blas device free error variable %s - %s.",
		   varname, QN_cublas_error_string(e)); 
    }
}

void
QN_MLP_BunchCudaVar::devfree_vf(const char* varname, const float* devptr)
{
    cublasStatus e;
    e = cublasFree((void *) devptr);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
	clog.error("cuda blas device free_vf error variable %s - %s.",
		   varname, QN_cublas_error_string(e)); 
    }
}

void
QN_MLP_BunchCudaVar::devfree_vi(const char* varname, const int* devptr)
{
    cublasStatus e;
    e = cublasFree((void *) devptr);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
	clog.error("cuda blas device free_vf error variable %s - %s.",
		   varname, QN_cublas_error_string(e)); 
    }
}

void
QN_MLP_BunchCudaVar::todev_vf_vf(const char* varname, int n, const float* from,
				 float* devto)
{
    cublasStatus e;

    e = cublasSetVector(n, sizeof(float), from, 1, devto, 1);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
	clog.error("cuda blas todev_vf_vf error variable %s - %s.",
		   varname, QN_cublas_error_string(e)); 
    }
    clog.log(QN_LOG_PER_BUNCH, "Copied %i floats to device variable \"%s\" at address %.8x\n", n, varname, devto);
}

void
QN_MLP_BunchCudaVar::fromdev_vf_vf(const char* varname, int n,
				   const float* devfrom, float* to)
{
    cublasStatus e;

    e = cublasGetVector(n, sizeof(float), devfrom, 1, to, 1);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
	clog.error("cuda blas fromdev_vf_vf error variable %s - %s.",
		   varname, QN_cublas_error_string(e)); 
    }
    clog.log(QN_LOG_PER_BUNCH, "Copied %i floats from device variable \"%s\" at address %.8x\n", n, varname, devfrom);
}

void
QN_MLP_BunchCudaVar::todev_vi_vi(const char* varname, int n,
				 const int* from, int* devto)
{
    cublasStatus e;

    e = cublasSetVector(n, sizeof(int), from, 1, devto, 1);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
	clog.error("cuda blas todev_vi_vi error variable %s - %s.",
		   varname, QN_cublas_error_string(e)); 
    }
    clog.log(QN_LOG_PER_BUNCH, "Copied %i ints to device variable \"%s\" at address %.8x\n", n, varname, devto);
}

void
QN_MLP_BunchCudaVar::fromdev_vi_vi(const char* varname, int n,
				   const int* devfrom, int* to)
{
    cublasStatus e;

    e = cublasGetVector(n, sizeof(int), devfrom, 1, to, 1);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
	clog.error("cuda blas fromdev_vi_vi error variable %s - %s.",
		   varname, QN_cublas_error_string(e)); 
    }
    clog.log(QN_LOG_PER_BUNCH, "Copied %i ints from device variable \"%s\" at address %.8x\n", n, varname, devfrom);
}




