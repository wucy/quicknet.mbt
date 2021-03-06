const char* QN_trn_rcsid =
    "$Header: /u/drspeech/repos/quicknet2/QN_trn.cc,v 1.24 2006/04/06 02:59:10 davidj Exp $";

// Routines for performing MLP trainings.

#include <QN_config.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "QN_types.h"
#include "QN_trn.h"
#include "QN_utils.h"
#include "QN_libc.h"
#include "QN_fltvec.h"
#include "QN_MLPWeightFile_RAP3.h"



//cw564 - mbt
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <stdio.h>
using std::ifstream;
using std::cin;
using std::cerr;
using std::cout;
using std::map;
using std::string;
using std::vector;
using std::endl;



//cw564 - mbt -- decode the modified lab buffer and generate right lab_buf and spkr_wgt per frame
void
QN_HardSentTrainer::convert_raw_lab_buf(QNUInt32 * lab_buf, float * * & spkr_wgt_buf, const int count)
{
    for (int i = 0; i < count; ++ i)
    {
        int segid = lab_buf[i] / mbt_params.lab_offset;
        lab_buf[i] %= mbt_params.lab_offset;

        spkr_wgt_buf[i] = mbt_params.seg2spkrwgt(segid);
    }

}

//cw564 - mbt -- concat the input
void
QN_HardSentTrainer::concat_raw_inp_buf(
        float * input_buf, 
        const int count,
        const int raw_fea_dim, 
        const int num_basis)
{
    float * cp_buf = new float[inp_buf_size];
    
    //TODO faster memcopy

    for (int i = 0; i < count * raw_fea_dim; ++ i)
    {
        cp_buf[i] = input_buf[i];
    }

    int pos = 0;
    for (int i = 0; i < count; ++ i)
    {
        for (int j = 0; j < num_basis; ++ j)
        {
            for (int k = 0; k < raw_fea_dim; ++ k)
            {
                input_buf[pos] = cp_buf[i * raw_fea_dim + k];
                pos ++;
            }
        }
    }
    delete [] cp_buf;
}

// "Hard training" object - trains using labels to indicate targets.

QN_HardSentTrainer::QN_HardSentTrainer(int a_debug, const char* a_dbgname,
				       int a_verbose,
				       QN_MLP* a_mlp,
				       QN_InFtrStream* a_train_ftr_str,
				       QN_InLabStream* a_train_lab_str,
				       QN_InFtrStream* a_cv_ftr_str,
				       QN_InLabStream* a_cv_lab_str,
				       QN_RateSchedule* a_lr_sched,
				       float a_targ_low, float a_targ_high,
				       const char* a_wlog_template,
				       QN_WeightFileType a_wfile_format,
				       const char* a_ckpt_template,
				       QN_WeightFileType a_ckpt_format,
				       unsigned long a_ckpt_secs,
				       size_t a_bunch_size,
				       int a_lastlab_reject,
				       float* a_lrscale,
                                       const MBT_Params * a_mbt_params //cw564 - mbt
                                       )
    : debug(a_debug),
      dbgname(a_dbgname),
      clog(a_debug, "QN_HardSentTrainer", a_dbgname),
      verbose(a_verbose),
      mlp(a_mlp),
      mlp_inps(mlp->size_layer((QN_LayerSelector) 0)),
      mlp_outs(mlp->size_layer((QN_LayerSelector) (mlp->num_layers()-1))),
      train_ftr_str(a_train_ftr_str),
      train_lab_str(a_train_lab_str),
      cv_ftr_str(a_cv_ftr_str),
      cv_lab_str(a_cv_lab_str),
      lr_sched(a_lr_sched),
      targ_low(a_targ_low),
      targ_high(a_targ_high),
      bunch_size(a_bunch_size),
      lastlab_reject(a_lastlab_reject ? 1 : 0),
      inp_buf_size(mlp_inps*bunch_size),
      out_buf_size(mlp_outs*bunch_size),
      targ_buf_size(out_buf_size),
      inp_buf(new float[inp_buf_size]),
      out_buf(new float[out_buf_size]),
      targ_buf(new float[targ_buf_size]),
      lab_buf(new QNUInt32[bunch_size]),
      spkr_wgt_buf(new float*[bunch_size]), //cw564 - mbt
      // Copy the template into a new char array.
      wlog_template(strcpy(new char[strlen(a_wlog_template)+1],
			   a_wlog_template)),
      wfile_format(a_wfile_format),
      ckpt_template(strcpy(new char[strlen(a_ckpt_template)+1],
			   a_ckpt_template)),
      ckpt_format(a_ckpt_format),
      ckpt_secs(a_ckpt_secs),
      last_ckpt_time(time(NULL)),
      pid(getpid()),
      mbt_params(*a_mbt_params) //cw564 - mbt
{
// Perform some checks of the input data.
    assert(bunch_size!=0);

    size_t i;
    // Copy across the lrscale vals
    if (a_lrscale!=NULL)
    {
	for (i=0; i<mlp->num_layers()-1; i++)
	    lrscale[i] = a_lrscale[i];
    }
    else
    {
	for (i=0; i<mlp->num_layers()-1; i++)
	    lrscale[i] = 1.0f;
    }
    // Check the input streams.
    /*if (train_ftr_str->num_ftrs()!=mlp_inps)
    {
	clog.error("The training feature stream has %lu features, the "
		   "MLP has %lu inputs.",
		   (unsigned long) train_ftr_str->num_ftrs(),
		   (unsigned long) mlp_inps);
    }
    if (cv_ftr_str->num_ftrs()!=mlp_inps)
    {
	clog.error("The CV feature stream has %lu features, the "
		   "MLP has %lu inputs.",
		   (unsigned long) cv_ftr_str->num_ftrs(),
		   (unsigned long) mlp_inps);
    }*/
    //cw564 - mbt - check input streams modified for mbt
    if (mlp_inps % train_ftr_str->num_ftrs() != 0)
    {
	clog.error("The training feature stream has %lu features, the "
		   "MLP has %lu inputs.",
		   (unsigned long) train_ftr_str->num_ftrs(),
		   (unsigned long) mlp_inps);
    }
    if (mlp_inps % cv_ftr_str->num_ftrs() != 0)
    {
	clog.error("The CV feature stream has %lu features, the "
		   "MLP has %lu inputs.",
		   (unsigned long) cv_ftr_str->num_ftrs(),
		   (unsigned long) mlp_inps);
    }

    if (train_lab_str->num_labs()!=1)
    {
	clog.error("The train label stream has %lu labels per frame but we "
		   "can only use 1.",
		   (unsigned long) train_lab_str->num_labs());
    }
    if (cv_lab_str->num_labs()!=1)
    {
	clog.error("The CV label stream has %lu labels per frame but we "
		   "can only use 1.",
		   (unsigned long) cv_lab_str->num_labs());
    }

    // Set up the weight logging stuff.
    if (QN_logfile_template_check(wlog_template)!=QN_OK)
    {
	clog.error("Invalid weight log file "
		 "template \'%s\'.", wlog_template);

    }
    // Set up the weight checkpointing stuff.
    if (QN_logfile_template_check(ckpt_template)!=QN_OK)
    {
	clog.error("Invalid ckpt file template \'%s\'.", ckpt_template);

    }
}

QN_HardSentTrainer::~QN_HardSentTrainer()
{
    delete[] ckpt_template;
    delete[] wlog_template;
    delete[] inp_buf;
    delete[] out_buf;
    delete[] targ_buf;
    delete[] lab_buf;

    delete [] spkr_wgt_buf; //cw564 - mbt
}


void
QN_HardSentTrainer::train(const char *lr_ctr, struct MapStruct *mapptr, bool if_mbt)	//cz277 - outmap	//cz277 - learn rate criterion
{
    double run_start_time;	// Time we started.
    double run_stop_time;	// Time we stopped.
    char timebuf[QN_TIMESTR_BUFLEN]; // Somewhere to put the current time.
    float percent_correct;	// Percentage correct on current test.
    float last_cv_error;	// Percentage error from last cross validation.
    size_t best_cv_epoch;	// Epoch of best cross validation error.
    size_t best_train_epoch;	// Epoch of best training error.
    float best_cv_error;	// Best CV error percentage.
    float best_train_error;	// Best train error percentage.
    char last_weightlog_filename[MAXPATHLEN];

    // Startup log messages.
    QN_timestr(timebuf, sizeof(timebuf));
    QN_OUTPUT("Training run start: %s.", timebuf);
    QN_OUTPUT("** ** ** ** ** ** ** ** ** ** ** ** ** **");
    run_start_time = QN_time();

    // Pre-training cross validation.
    QN_timestr(timebuf, sizeof(timebuf));
    QN_OUTPUT("Pre-run cross validation started: %s.", timebuf);
    percent_correct = cv_epoch(lr_ctr, mapptr);	//cz277 - device select
    QN_timestr(timebuf, sizeof(timebuf));
    if (verbose)
	QN_OUTPUT("Pre-run cross validation finished: %s.", timebuf);
    //exit(0);
    // Note: to prevent the initial weights being saved as the best weights,
    // we assume the cross validation had abysmal results.  Even if the
    // training makes things worse, the change might be useful and we do not
    // want the resulting weights to be from the initialization file.
    last_cv_error = 100.0f;
    best_cv_error = 100.0f;
    best_train_error = 100.0f;
    best_cv_epoch = 0;
    best_train_epoch = 0;
    learn_rate = lr_sched->get_rate();
    epoch = 1;			// Epochs are numbered starting from 1.
    
    //cv_ftr_str->rewind();
    //cv_lab_str->rewind();
    //train_ftr_str->rewind();
    //train_lab_str->rewind();

    //cw564 - mbt -- TODO
    //lambda_epoch(lr_ctr, mapptr);
    //exit(0);

    while (learn_rate!=0.0f)	// Iterate over all epochs.
    {
	float train_error;	// Percentage error on current training.
	float cv_error;		// Percentage error on current CV.
	
	// Epoch startup status.
	QN_OUTPUT("** ** ** ** ** ** ** ** ** ** ** ** ** **");
	QN_OUTPUT("New epoch: epoch %i, learn rate %f.", epoch, learn_rate);
	QN_timestr(timebuf, sizeof(timebuf));
	QN_OUTPUT("Epoch started: %s.", timebuf);

	// Training phase.
	set_learnrate();	// Set the learning rate 
	train_ftr_str->rewind();
	train_lab_str->rewind();
	percent_correct = train_epoch(mapptr);	//cz277 - outmap
	train_error = 100.0f - percent_correct;
	//exit(0);
    if (train_error < best_train_error)
	{
	    best_train_error = train_error;
	    best_train_epoch = epoch;
	}
	QN_timestr(timebuf, sizeof(timebuf));
	if (verbose)
	    QN_OUTPUT("Training finished/CV started: %s.", timebuf);

	// Cross validation phase.
	cv_ftr_str->rewind();
	cv_lab_str->rewind();


	percent_correct = cv_epoch(lr_ctr, mapptr);	//cz277 - outmap	//cz277 - learn rate criterion
	cv_error = 100.0f - percent_correct;

	// Keeping track of how well we are doing.
	if ((strcmp(lr_sched->get_state(), "list") != 0) && (cv_error > last_cv_error))	//cz277 - list restore
	{
	    QN_OUTPUT("Weight log: Restoring previous weights from "
	    "`%s\'.", last_weightlog_filename);
	    QN_readwrite_weights(debug, dbgname, *mlp,
				 last_weightlog_filename, wfile_format,
				 QN_READ, NULL, NULL, mbt_params.num_basis);
	}
	else
	{
	    int ec;		// Error code.

	    ec = QN_logfile_template_map(wlog_template,
					 last_weightlog_filename, MAXPATHLEN,
					 epoch, pid);
	    if (ec!=QN_OK)
	    {
		clog.error("failed to build weight log file name from "
			   "template \'%s\'.", wlog_template);
	    }
	    QN_OUTPUT("Weight log: Saving weights to `%s\'.",
		      last_weightlog_filename);
	    QN_readwrite_weights(debug,dbgname, *mlp,
	    			 last_weightlog_filename, wfile_format, QN_WRITE, NULL, NULL, mbt_params.num_basis);
	    last_cv_error = cv_error;
	    best_cv_error = cv_error;
	    best_cv_epoch = epoch;
	}

	// Epoch end status.
	QN_timestr(timebuf, sizeof(timebuf));
	if (verbose)
	    QN_OUTPUT("Epoch finished: %s.", timebuf);

	// On to next epoch.
	learn_rate = lr_sched->next_rate(cv_error);
	epoch++;
    }

    // Wind down log messages.
    QN_OUTPUT("** ** ** ** ** ** ** ** ** ** ** ** ** **");
    QN_OUTPUT("Best CV accuracy: %.2f%% correct in epoch %i.",
	    100.0 - best_cv_error, (int) best_cv_epoch);
    QN_OUTPUT("Best train accuracy: %.2f%% correct in epoch %i.",
	    100.0 - best_train_error, (int) best_train_epoch);
    run_stop_time = QN_time();
    QN_timestr(timebuf, sizeof(timebuf));
    QN_OUTPUT("Training run stop: %s.", timebuf);
    double total_time = run_stop_time - run_start_time;
    int hours = ((int) total_time) / 3600;
    int mins = (((int) total_time) / 60) % 60;
    int secs = ((int) total_time) % 60;
    QN_OUTPUT("Training time: %.2f secs (%i hours, %i mins, %i secs).",
	      total_time, hours, mins, secs);
    QN_OUTPUT("** ** ** ** ** ** ** ** ** ** ** ** ** **");
}

//cw564 - mbt - lambda_epoch
double
QN_HardSentTrainer::lambda_epoch(const char *lr_ctr, struct MapStruct *mapptr, bool if_mbt)
{
    vector< vector< float > > train_last_y;
    vector< vector< float > > train_out;
    vector< int > train_lab;

    vector< vector< float > > W;
    vector< float > b;

    size_t correct_02e = 0;
    size_t total_02e = 0;

    size_t ftr_count;		// Count of feature frames read.
    size_t lab_count;		// Count of label frames read.
    size_t total_frames = 0;	// Total frames read this phase.
    size_t correct_frames = 0;	// Number of correct frames this phase.
    size_t reject_frames = 0;	// Numer of frames we reject.
    size_t current_segno;	// Current segment number.
    double start_secs;		// Exact time we started.
    double stop_secs;		// Exact time we stopped.
    size_t i;			// Local counter.

    //cz277 - outmap
    size_t map_correct_frames = 0;

    current_segno = 0;
    ftr_count = 0;		// Pretend that previous read hit end of seg.
    start_secs = QN_time();
   
    // Iterate over all input segments.
    // Note that, at this stage, an input segment is _not_ typically a
    // sentence, but a buffer full of sentences (known as a "fill" in old
    // QuickNet code).
    while (1)
    {
	if (ftr_count<bunch_size) // Check if at end of segment.
	{
	    QN_SegID ftr_segid;	// Segment ID from input stream.
	    QN_SegID lab_segid;	// Segment ID from label stream.

	    ftr_segid = cv_ftr_str->nextseg();
	    lab_segid = cv_lab_str->nextseg();
	    assert(ftr_segid==lab_segid);
	    if (ftr_segid==QN_SEGID_BAD)
		break;
	    else
	    {
		current_segno++;
	    }
	}

	// Get the data to pass on to the net.
	ftr_count = cv_ftr_str->read_ftrs(bunch_size, inp_buf);
	lab_count = cv_lab_str->read_labs(bunch_size, lab_buf);
       


    if (ftr_count!=lab_count)
	{
	    clog.error("Feature and label streams have different segment "
		       "lengths in cross validation.");
	}
        
    //cw564 - mbt
    float * raw_lab_buf = new float[lab_count];
    for (int i = 0; i < lab_count; ++ i) raw_lab_buf[i] = lab_buf[i];
       
    //cw564 - mbt
    //error!!
    concat_raw_inp_buf(inp_buf, ftr_count, cv_ftr_str->num_ftrs(), mbt_params.num_basis);
    convert_raw_lab_buf(lab_buf, spkr_wgt_buf, lab_count);

	// Do the forward pass.
	mlp->forward(ftr_count, inp_buf, out_buf, spkr_wgt_buf, mbt_params.num_basis);
    cout << ftr_count << endl; 
    float * last_y = mlp->Last_y();
	float dim_per_item = mlp->size_layer((QN_LayerSelector) (mlp->num_layers()-2));
    float dim_per_out = mlp->size_layer((QN_LayerSelector) (mlp->num_layers()-1));
    int nowptr = 0;
    int out_ptr = 0;
    for (int i = 0; i < ftr_count; ++ i)
    {
        int now_lab = raw_lab_buf[i];
        vector< float > vect;
        for (int j = 0; j < dim_per_item; ++ j)
        {
            vect.push_back(last_y[nowptr]);
            nowptr ++;
        }
        vector< float > out;
        for (int j = 0; j < dim_per_out; ++ j)
        {
            out.push_back(out_buf[out_ptr]);
            out_ptr ++;
        } 
        train_last_y.push_back(vect);
        train_out.push_back(out);
        train_lab.push_back(now_lab);
    }

	// Analyze the output of the net.
	float* out_buf_ptr = out_buf; // Current output frame.
	QNUInt32* lab_buf_ptr = lab_buf; // Current label.
	QNUInt32 net_label;		// Label chosen by the net.
	QNUInt32 cv_label;		// Label in cv stream.
	for (int i=0; i<ftr_count; i++)
	{
            if (mbt_params.seg2spkr[raw_lab_buf[i] / 10000] == "02e")
                total_02e ++;

	    net_label = qn_imax_vf_u(mlp_outs, out_buf_ptr);
	    cv_label = *lab_buf_ptr;
	    if (cv_label >= (mlp_outs + lastlab_reject))
	    {
		clog.error("Label value of %lu in CV stream is "
			   "larger than the number of output units.",
			   (unsigned long) cv_label);
	    }
	    if (lastlab_reject && cv_label==mlp_outs)
		reject_frames++;
	    else if (net_label==cv_label) {
                if (mbt_params.seg2spkr[raw_lab_buf[i] / 10000] == "02e")
                {
                    correct_02e ++;
                }
		correct_frames++;
            }
	    out_buf_ptr += mlp_outs;
	    lab_buf_ptr++;
	}
	total_frames += ftr_count;

        //cz277 - outmap
    if (mapptr->opt > 0) 
	{
	    out_buf_ptr = out_buf;
        lab_buf_ptr = lab_buf;
        for (i=0; i<ftr_count; i++)
        {
		    if (mapptr->opt == 1) {	//max
                net_label = mapptr->srclist[qn_imax_vf_u(mlp_outs, out_buf_ptr)];
		    } else {	//sum
		        memset(mapptr->tgtlist, 0.0, mapptr->tgtdim * sizeof(int));
		        //printf("v == %f ", mapptr->tgtlist[0]);
		        for (int j = 0; j != mapptr->srcdim; ++j) {
			    //	printf("\t%f\n", out_buf_ptr[j]);
		        mapptr->tgtlist[mapptr->srclist[j]] += out_buf_ptr[j];
		        }
		        if (mapptr->opt == 3) {	//avg
			        for (int j = 0; j != mapptr->tgtdim; ++j) {
			            mapptr->tgtlist[j] /= mapptr->tgtcnts[j];
			        }
		        }
		        net_label = qn_imax_vf_u(mapptr->tgtdim, mapptr->tgtlist);
		    }	
            cv_label = mapptr->srclist[*lab_buf_ptr];

            if (net_label==cv_label)
                map_correct_frames++;
            out_buf_ptr += mlp_outs;
            lab_buf_ptr++;
	    }

    }
        

    delete [] raw_lab_buf;

    }
    stop_secs = QN_time();

    //cerr << total_02e << ' ' << correct_02e << endl;
   
    //exit(0);
    map<string, std::ofstream *> ofsfealab_map;
    map<string, std::ofstream *> ofsout_map;

    for (map<string, float *>::iterator it = mbt_params.spkr2wgt.begin(); 
            it != mbt_params.spkr2wgt.end(); ++ it)
    {
        cout << it->first << endl;
        string ofsfealabname = "workdir/" + it->first + "fealab";
        string ofsoutname = "workdir/" + it->first + "out";
        
        std::ofstream * ofs = new std::ofstream(ofsfealabname.c_str());
        ofsfealab_map[it->first] = ofs;

        std::ofstream * ofs_out = new std::ofstream(ofsoutname.c_str());
        ofsout_map[it->first] = ofs_out;
    }

    for (int i = 0; i < train_lab.size(); ++ i)
    {
        string myid = mbt_params.seg2spkr[train_lab[i] / 10000];
        //girl:02e boy:02b cv:40l
        //if (myid != "40l") continue;
        std::ofstream & ofs = *(ofsfealab_map[myid]);
        ofs << train_lab[i] % 10000;
        for (int j = 0; j < train_last_y[i].size(); ++ j)
        {
            ofs << ' ' << train_last_y[i][j];
        }
        ofs << endl;
        std::ofstream & outofs = *(ofsout_map[myid]);
        for (int j = 0; j < train_out[i].size(); ++ j)
        {
            outofs << train_out[i][j] << ' ';
        }
        outofs << endl;
    }

    for (map<string, std::ofstream * >::iterator it = ofsfealab_map.begin(); it != ofsfealab_map.end(); ++ it)
    {
        it->second->close();
        ofsfealab_map[it->first]->close();
    }

    exit(0);

    double total_secs = stop_secs - start_secs;
    size_t unreject_frames = total_frames - reject_frames;
    double percent = 100.0 * (double) correct_frames / (double) unreject_frames; 

    QN_OUTPUT("CV speed: %.2f MCPS, %.1f presentations/sec.",
	      QN_secs_to_MCPS(stop_secs-start_secs, total_frames, *mlp),
	      (double) total_frames / total_secs);
    QN_OUTPUT("CV accuracy:  %lu right out of %lu, %.2f%% correct.",
	      (unsigned long) correct_frames, (unsigned long) unreject_frames,
	      percent);

    //cz277 - outmap
    if (mapptr->opt > 0) {
	QN_OUTPUT("Mapped CV accuracy:  %lu right out of %lu, %.2f%% correct.",
              (unsigned long) map_correct_frames, (unsigned long) unreject_frames,
              100.0 * (double) map_correct_frames / (double) unreject_frames);
    }

    //cz277 - learn rate criterion
    if (strcmp(lr_ctr, "mapped_cv") == 0) {
        percent = 100.0 * (double) map_correct_frames / (double) unreject_frames;
    }

    if (lastlab_reject)
    {
	double percent_reject = 100.0
		* (double)reject_frames / (double) total_frames;
	QN_OUTPUT("CV reject frames: %lu of total %lu frames rejected, "
		  "%.2f%% rejected.",
		  (unsigned long) reject_frames,
		  (unsigned long) total_frames,
		  percent_reject);
    }

    return percent; 
}


// Note - cross validation is less demanding on the facilities that
// must be provided by the stream.
double
QN_HardSentTrainer::cv_epoch(const char *lr_ctr, struct MapStruct *mapptr, bool if_mbt)	//cz277 - outmap	//cz277 - learn rate criterion
{
    size_t ftr_count;		// Count of feature frames read.
    size_t lab_count;		// Count of label frames read.
    size_t total_frames = 0;	// Total frames read this phase.
    size_t correct_frames = 0;	// Number of correct frames this phase.
    size_t reject_frames = 0;	// Numer of frames we reject.
    size_t current_segno;	// Current segment number.
    double start_secs;		// Exact time we started.
    double stop_secs;		// Exact time we stopped.
    size_t i;			// Local counter.

    //cz277 - outmap
    size_t map_correct_frames = 0;

    current_segno = 0;
    ftr_count = 0;		// Pretend that previous read hit end of seg.
    start_secs = QN_time();
   
    // Iterate over all input segments.
    // Note that, at this stage, an input segment is _not_ typically a
    // sentence, but a buffer full of sentences (known as a "fill" in old
    // QuickNet code).
    while (1)
    {
	if (ftr_count<bunch_size) // Check if at end of segment.
	{
	    QN_SegID ftr_segid;	// Segment ID from input stream.
	    QN_SegID lab_segid;	// Segment ID from label stream.

	    ftr_segid = cv_ftr_str->nextseg();
	    lab_segid = cv_lab_str->nextseg();
	    assert(ftr_segid==lab_segid);
	    if (ftr_segid==QN_SEGID_BAD)
		break;
	    else
	    {
		current_segno++;
	    }
	}

	// Get the data to pass on to the net.
	ftr_count = cv_ftr_str->read_ftrs(bunch_size, inp_buf);
	lab_count = cv_lab_str->read_labs(bunch_size, lab_buf);
        

        if (ftr_count!=lab_count)
	{
	    clog.error("Feature and label streams have different segment "
		       "lengths in cross validation.");
	}
        
        //cw564 - mbt
        //cerr << lab_buf[0] << endl;
        concat_raw_inp_buf(inp_buf, ftr_count, cv_ftr_str->num_ftrs(), mbt_params.num_basis);
        convert_raw_lab_buf(lab_buf, spkr_wgt_buf, lab_count);
        //cerr << lab_buf[0] << '\t' << spkr_wgt_buf[0]  << '\t' 
        //    << spkr_wgt_buf[0][0] << '\t' << spkr_wgt_buf[0][1] << endl; exit(0);

	// Do the forward pass.
        //cerr << mbt_params.num_basis << endl;
	mlp->forward(ftr_count, inp_buf, out_buf, spkr_wgt_buf, mbt_params.num_basis);

	

	// Analyze the output of the net.
	float* out_buf_ptr = out_buf; // Current output frame.
	QNUInt32* lab_buf_ptr = lab_buf; // Current label.
	QNUInt32 net_label;		// Label chosen by the net.
	QNUInt32 cv_label;		// Label in cv stream.
	for (i=0; i<ftr_count; i++)
	{
	    net_label = qn_imax_vf_u(mlp_outs, out_buf_ptr);
	    cv_label = *lab_buf_ptr;
	    if (cv_label >= (mlp_outs + lastlab_reject))
	    {
		clog.error("Label value of %lu in CV stream is "
			   "larger than the number of output units.",
			   (unsigned long) cv_label);
	    }
	    if (lastlab_reject && cv_label==mlp_outs)
		reject_frames++;
	    else if (net_label==cv_label)
		correct_frames++;
	    out_buf_ptr += mlp_outs;
	    lab_buf_ptr++;
	}
	total_frames += ftr_count;

        //cz277 - outmap
        if (mapptr->opt > 0) 
	{
	    out_buf_ptr = out_buf;
            lab_buf_ptr = lab_buf;
            for (i=0; i<ftr_count; i++)
            {
		if (mapptr->opt == 1) {	//max
                    net_label = mapptr->srclist[qn_imax_vf_u(mlp_outs, out_buf_ptr)];
		} else {	//sum
		    memset(mapptr->tgtlist, 0.0, mapptr->tgtdim * sizeof(int));
		    //printf("v == %f ", mapptr->tgtlist[0]);
		    for (int j = 0; j != mapptr->srcdim; ++j) {
			//	printf("\t%f\n", out_buf_ptr[j]);
		        mapptr->tgtlist[mapptr->srclist[j]] += out_buf_ptr[j];
		    }
		    if (mapptr->opt == 3) {	//avg
			for (int j = 0; j != mapptr->tgtdim; ++j) {
			    mapptr->tgtlist[j] /= mapptr->tgtcnts[j];
			}
		    }

		    net_label = qn_imax_vf_u(mapptr->tgtdim, mapptr->tgtlist);
		}	
                cv_label = mapptr->srclist[*lab_buf_ptr];

                if (net_label==cv_label)
                    map_correct_frames++;
                out_buf_ptr += mlp_outs;
                lab_buf_ptr++;
	    }
        }

    }
    stop_secs = QN_time();
    double total_secs = stop_secs - start_secs;
    size_t unreject_frames = total_frames - reject_frames;
    double percent = 100.0 * (double) correct_frames
	/ (double) unreject_frames; 

    QN_OUTPUT("CV speed: %.2f MCPS, %.1f presentations/sec.",
	      QN_secs_to_MCPS(stop_secs-start_secs, total_frames, *mlp),
	      (double) total_frames / total_secs);
    QN_OUTPUT("CV accuracy:  %lu right out of %lu, %.2f%% correct.",
	      (unsigned long) correct_frames, (unsigned long) unreject_frames,
	      percent);

    //cz277 - outmap
    if (mapptr->opt > 0) {
	QN_OUTPUT("Mapped CV accuracy:  %lu right out of %lu, %.2f%% correct.",
              (unsigned long) map_correct_frames, (unsigned long) unreject_frames,
              100.0 * (double) map_correct_frames / (double) unreject_frames);
    }

    //cz277 - learn rate criterion
    if (strcmp(lr_ctr, "mapped_cv") == 0) {
        percent = 100.0 * (double) map_correct_frames / (double) unreject_frames;
    }

    if (lastlab_reject)
    {
	double percent_reject = 100.0
		* (double)reject_frames / (double) total_frames;
	QN_OUTPUT("CV reject frames: %lu of total %lu frames rejected, "
		  "%.2f%% rejected.",
		  (unsigned long) reject_frames,
		  (unsigned long) total_frames,
		  percent_reject);
    }

    return percent; 
}

double
QN_HardSentTrainer::train_epoch(struct MapStruct *mapptr, bool if_mbt) //cw564 - mbt
{
    size_t ftr_count;		// Count of feature frames read.
    size_t lab_count;		// Count of label frames read.
    size_t trn_count;		// Count of frames to train on.
    size_t total_frames = 0;	// Total frames read this phase.
    size_t seg_frames = 0;      // Number of frames in this segment.
    size_t reject_frames = 0;	// Total number of rejects this phase.
    size_t unreject_frames;	// Number of not rejected frames.
    size_t correct_frames = 0;	// Number of correct frames this phase.
    size_t total_segs;		// Number of segments in streams.
    size_t current_segno;	// Current segment number.
    time_t current_time;	// Current time.
    double start_secs;		// Exact time we started.
    double stop_secs;		// Exact time we stopped.
    double total_secs;		// Total time.
    double percent;
    size_t i;			// Local counter.

    assert(bunch_size!=0);

    total_segs = train_ftr_str->num_segs();
    current_segno = 0;
    ftr_count = 0;		// Pretend that previous read hit end of seg.
    start_secs = QN_time();
   
    //cz277 - outmap
    size_t map_correct_frames = 0;
 
    // Iterate over all input segments.
    // Note that, at this stage, an input segment is _not_ typically a
    // sentence, but a buffer full of sentences (known as a "fill" in old
    // QuickNet code).
    while (1)
    {
	if (ftr_count<bunch_size) // Check if at end of segment.
	{
	    QN_SegID ftr_segid;	// Segment ID from input stream.
	    QN_SegID lab_segid;	// Segment ID from label stream.
	    
	    // update the learning rate, for those schedules that care
	    float nlearn_rate=lr_sched->trained_on_nsamples(seg_frames);
	    if (nlearn_rate != learn_rate) {
	      learn_rate=nlearn_rate;
	      set_learnrate();  // Pass the info along to the MLP
	      if (verbose) 
		QN_OUTPUT("learning rate set to %.6f after %d samples read",learn_rate,seg_frames);
	    }
	    seg_frames=0;
	    
	    ftr_segid = train_ftr_str->nextseg();
	    lab_segid = train_lab_str->nextseg();
	    assert(ftr_segid==lab_segid);
	    if (ftr_segid==QN_SEGID_BAD)
		break;
	    else
	    {
		if (verbose)
		{
		    QN_OUTPUT("Starting chunk %lu of %lu containing "
			      "%lu frames...",
			      (unsigned long) (current_segno+1),
			      (unsigned long) total_segs,
			      train_ftr_str->num_frames(current_segno));
		}
		current_segno++;
	    }
	}

	// Get the data to pass on to the net.
	ftr_count = train_ftr_str->read_ftrs(bunch_size, inp_buf);
	lab_count = train_lab_str->read_labs(bunch_size, lab_buf);

	if (ftr_count!=lab_count)
	{
	    clog.error("Feature and label streams have different segment "
		       "lengths in cross validation.");
	}
        
        //cw564 - mbt
        convert_raw_lab_buf(lab_buf, spkr_wgt_buf, lab_count);
        concat_raw_inp_buf(inp_buf, ftr_count, cv_ftr_str->num_ftrs(), mbt_params.num_basis);

	// Check that the label stream is good and build up the target vector.
	qn_copy_f_vf(ftr_count*mlp_outs, targ_low, targ_buf);
	float* targ_buf_ptr = targ_buf;	// Target values put here
	QNUInt32* lab_buf_ptr = lab_buf; // Labels taken from here
	QNUInt32 lab;		// Label to train to
	if (lastlab_reject)
	{
	    // Set up the input and target vectors taking account of rejects
	    // Note that we compress the input features in place.
	    float* inp_buf_ptr = inp_buf; // Get all frames from here
	    float* unrej_buf_ptr = inp_buf; // Put unrejected frames here

	    trn_count = 0;
	    for (i=0; i<lab_count; i++)
	    {

		lab = *lab_buf_ptr;
		if (lab>mlp_outs)
		{
		    clog.error("Label in train stream is larger than the "
			       "number of output units.");
		}
		else if (lab==mlp_outs)
		{
		    // Rejected frame.
		    reject_frames++;
		}
		else
		{
		    // Unrejected frame.
		    trn_count++;
		    targ_buf_ptr[lab] = targ_high;
		    qn_copy_vf_vf(mlp_inps, inp_buf_ptr, unrej_buf_ptr);
		    targ_buf_ptr += mlp_outs;
		    unrej_buf_ptr += mlp_inps;
		}
		inp_buf_ptr+= mlp_inps;
		lab_buf_ptr++;
	    }
	}
	else
	{
	    // Here we set up the target vector without reject labels
	    for (i=0; i<lab_count; i++)
	    {
		lab = *lab_buf_ptr;
		if (lab>=mlp_outs)
		{
		    clog.error("Label in train stream is larger than the "
			       "number of output units.");
		}
		targ_buf_ptr[lab] = targ_high;
		lab_buf_ptr++;
		targ_buf_ptr += mlp_outs;
	    }
	    trn_count = lab_count;
	}

	// Do the training.
	mlp->train(trn_count, inp_buf, targ_buf, out_buf, spkr_wgt_buf, mbt_params.num_basis);
         


	// Analyze the output of the net.
	float* out_buf_ptr = out_buf; // Current output frame.
	lab_buf_ptr = lab_buf;	// Current label.
	QNUInt32 net_label;		// Label chosen by the net.
	for (i=0; i<trn_count; i++)
	{
	    net_label = qn_imax_vf_u(mlp_outs, out_buf_ptr);
	    while (*lab_buf_ptr==mlp_outs)
		lab_buf_ptr++; // skip reject labels
	    if (net_label==*lab_buf_ptr)
		correct_frames++;
	    out_buf_ptr += mlp_outs;
	    lab_buf_ptr++;
	}

	//cz277 - outmap
	if (mapptr->opt > 0)
	{
	    out_buf_ptr = out_buf;
	    lab_buf_ptr = lab_buf;
	    for (i=0; i<trn_count; i++)
	    {
	        if (mapptr->opt == 1) { //max
	            net_label = mapptr->srclist[qn_imax_vf_u(mlp_outs, out_buf_ptr)];
	        } else {        //sum
	            memset(mapptr->tgtlist, 0.0, mapptr->tgtdim * sizeof(int));
	            for (int j = 0; j != mapptr->srcdim; ++j) {
	                mapptr->tgtlist[mapptr->srclist[j]] += out_buf_ptr[j];
	            }
	            if (mapptr->opt == 3) {     //avg
	                for (int j = 0; j != mapptr->tgtdim; ++j) {
	                    mapptr->tgtlist[j] /= mapptr->tgtcnts[j];
	                }
	            }
	
	            net_label = qn_imax_vf_u(mapptr->tgtdim, mapptr->tgtlist);
	        }
	        QNUInt32 mapped_label = mapptr->srclist[*lab_buf_ptr];
	
		while (*lab_buf_ptr==mlp_outs)
                    lab_buf_ptr++; // skip reject labels

	        if (net_label==mapped_label)
	            map_correct_frames++;
	        out_buf_ptr += mlp_outs;
	        lab_buf_ptr++;
	    }
	}

	total_frames += ftr_count;
	seg_frames += ftr_count;

	// Checkpoint if necessary.
	current_time = time(NULL);
	if (ckpt_secs!=0 && (current_time > (last_ckpt_time + ckpt_secs)))
	{
	    last_ckpt_time = current_time;
	    checkpoint_weights();
	}
    }



    stop_secs = QN_time();
    total_secs = stop_secs - start_secs;
    unreject_frames = total_frames - reject_frames;
    percent = 100.0 * (double) correct_frames / (double) unreject_frames;

    QN_OUTPUT("Train speed: %.2f MCUPS, %.1f presentations/sec.",
	      QN_secs_to_MCPS(stop_secs-start_secs, unreject_frames, *mlp),
	      (double) unreject_frames / total_secs);
    QN_OUTPUT("Train accuracy:  %lu right out of %lu, %.2f%% correct.",
	      (unsigned long) correct_frames, (unsigned long) unreject_frames,
	      percent);

    //cz277 - outmap
    if (mapptr->opt > 0) {
        QN_OUTPUT("Mapped Train accuracy:  %lu right out of %lu, %.2f%% correct.",
              (unsigned long) map_correct_frames, (unsigned long) unreject_frames,
              100.0 * (double) map_correct_frames / (double) unreject_frames);
    }

    if (lastlab_reject)
    {
	double percent_reject;

	percent_reject = 100.0 * (double)reject_frames / (double)total_frames;
	QN_OUTPUT("Train reject frames: %lu of total %lu frames rejected, "
		  "%.2f%% rejected.",
		  (unsigned long) reject_frames,
		  (unsigned long) total_frames,
		  percent_reject);
    }

    return percent; 
}

// Set the learning rate for the net according to the learn_rate variable.
void
QN_HardSentTrainer::set_learnrate()
{
    size_t i;
    float scale;		// Learnrate scaler.

    for (i=0; i<mlp->num_sections(); i++)
    {
	scale = lrscale[i/2];
	mlp->set_learnrate((QN_SectionSelector) i, learn_rate*scale);
    }
}

void
QN_HardSentTrainer::checkpoint_weights()
{
    int ec;
    char ckpt_filename[MAXPATHLEN]; // Checkpoint filename.
    
    ec = QN_logfile_template_map(ckpt_template,
				 ckpt_filename, MAXPATHLEN,
				 epoch, pid);
    if (ec!=QN_OK)
    {
	clog.error("failed to build ckpt weight file name from "
		   "template \'%s\'.", ckpt_template);
    }
    QN_OUTPUT("Checkpoint: Saving weights to `%s\'.",
	      ckpt_filename);
    QN_readwrite_weights(debug,dbgname, *mlp,
			 ckpt_filename, ckpt_format, QN_WRITE, NULL, NULL, mbt_params.num_basis);

}


////////////////////////////////////////////////////////////////
// "Soft training" object - trains using continuous targets.

QN_SoftSentTrainer::QN_SoftSentTrainer(int a_debug, const char* a_dbgname,
				       int a_verbose,
				       QN_MLP* a_mlp,
				       QN_InFtrStream* a_train_ftr_str,
				       QN_InFtrStream* a_train_targ_str,
				       QN_InFtrStream* a_cv_ftr_str,
				       QN_InFtrStream* a_cv_targ_str,
				       QN_RateSchedule* a_lr_sched,
				       float a_targ_low, float a_targ_high,
				       const char* a_wlog_template,
				       QN_WeightFileType a_wfile_format,
				       const char* a_ckpt_template,
				       QN_WeightFileType a_ckpt_format,
				       unsigned long a_ckpt_secs,
				       size_t a_bunch_size,
				       float* a_lrscale)
    : clog(a_debug, "QN_SoftSentTrainer", a_dbgname),
      verbose(a_verbose),
      mlp(a_mlp),
      mlp_inps(mlp->size_layer((QN_LayerSelector) 0)),
      mlp_outs(mlp->size_layer((QN_LayerSelector) (mlp->num_layers()-1))),
      train_ftr_str(a_train_ftr_str),
      train_targ_str(a_train_targ_str),
      cv_ftr_str(a_cv_ftr_str),
      cv_targ_str(a_cv_targ_str),
      lr_sched(a_lr_sched),
      targ_low(a_targ_low),
      targ_high(a_targ_high),
      bunch_size(a_bunch_size),
      inp_buf_size(mlp_inps*bunch_size),
      out_buf_size(mlp_outs*bunch_size),
      targ_buf_size(out_buf_size),
      inp_buf(new float[inp_buf_size]),
      out_buf(new float[out_buf_size]),
      targ_buf(new float[targ_buf_size]),
      // Copy the template into a new char array.
      wlog_template(strcpy(new char[strlen(a_wlog_template)+1],
			   a_wlog_template)),
      wfile_format(a_wfile_format),
      ckpt_template(strcpy(new char[strlen(a_ckpt_template)+1],
			   a_ckpt_template)),
      ckpt_format(a_ckpt_format),
      ckpt_secs(a_ckpt_secs),
      last_ckpt_time(time(NULL)),
      pid(getpid())
{
// Perform some checks of the input data.
    assert(bunch_size!=0);

    size_t i;
    // Copy across the lrscale vals
    if (a_lrscale!=NULL)
    {
	for (i=0; i<mlp->num_layers()-1; i++)
	    lrscale[i] = a_lrscale[i];
    }
    else
    {
	for (i=0; i<mlp->num_layers()-1; i++)
	    lrscale[i] = 1.0f;
    }
    // Check the input streams.
    if (train_ftr_str->num_ftrs()!=mlp_inps)
    {
	clog.error("The training feature stream has %lu features, the "
		   "MLP has %lu inputs.",
		   (unsigned long) train_ftr_str->num_ftrs(),
		   (unsigned long) mlp_inps);
    }
    if (cv_ftr_str->num_ftrs()!=mlp_inps)
    {
	clog.error("The CV feature stream has %lu features, the "
		   "MLP has %lu inputs.",
		   (unsigned long) cv_ftr_str->num_ftrs(),
		   (unsigned long) mlp_inps);
    }
    if (train_targ_str->num_ftrs()!=mlp_outs)
    {
	clog.error("The training target stream has %lu features, the "
		   "MLP has %lu outputs.",
		   (unsigned long) train_targ_str->num_ftrs(),
		   (unsigned long) mlp_outs);
    }
    if (cv_targ_str->num_ftrs()!=mlp_outs)
    {
	clog.error("The CV target stream has %lu features, the "
		   "MLP has %lu outputs.",
		   (unsigned long) cv_targ_str->num_ftrs(),
		   (unsigned long) mlp_outs);
    }

    // Set up the weight logging stuff.
    if (QN_logfile_template_check(wlog_template)!=QN_OK)
    {
	clog.error("Invalid weight log file "
		 "template \'%s\'.", wlog_template);

    }

    // Set up the weight checkpointing stuff.
    if (QN_logfile_template_check(ckpt_template)!=QN_OK)
    {
	clog.error("Invalid ckpt file template \'%s\'.", ckpt_template);

    }
}

QN_SoftSentTrainer::~QN_SoftSentTrainer()
{
    delete[] ckpt_template;
    delete[] wlog_template;
    delete[] inp_buf;
    delete[] out_buf;
    delete[] targ_buf;
}

void
QN_SoftSentTrainer::train()
{
    double run_start_time;	// Time we started.
    double run_stop_time;	// Time we stopped.
    char timebuf[QN_TIMESTR_BUFLEN]; // Somewhere to put the current time.
    float percent_correct;	// Percentage correct on current test.
    float last_cv_error;	// Percentage error from last cross validation.
    size_t best_cv_epoch;	// Epoch of best cross validation error.
    size_t best_train_epoch;	// Epoch of best training error.
    float best_cv_error;	// Best CV error percentage.
    float best_train_error;	// Best train error percentage.
    char last_weightlog_filename[MAXPATHLEN];

    // Startup log messages.
    QN_timestr(timebuf, sizeof(timebuf));
    QN_OUTPUT("Training run start: %s.", timebuf);
    QN_OUTPUT("** ** ** ** ** ** ** ** ** ** ** ** ** **");
    run_start_time = QN_time();

    // Pre-training cross validation.
    QN_timestr(timebuf, sizeof(timebuf));
    QN_OUTPUT("Pre-run cross validation started: %s.", timebuf);
    percent_correct = cv_epoch();
    QN_timestr(timebuf, sizeof(timebuf));
    if (verbose)
	QN_OUTPUT("Pre-run cross validation finished: %s.", timebuf);

    // Note: to prevent the initial weights being saved as the best weights,
    // we assume the cross validation had abysmal results.  Even if the
    // training makes things worse, the change might be useful and we do not
    // want the resulting weights to be from the initialization file.
    last_cv_error = 100.0f;
    best_cv_error = 100.0f;
    best_train_error = 100.0f;
    best_cv_epoch = 0;
    best_train_epoch = 0;
    learn_rate = lr_sched->get_rate();
    epoch = 1;			// Epochs are numbered starting from 1.

    while (learn_rate!=0.0f)	// Iterate over all epochs.
    {
	float train_error;	// Percentage error on current training.
	float cv_error;		// Percentage error on current CV.
	
	// Epoch startup status.
	QN_OUTPUT("** ** ** ** ** ** ** ** ** ** ** ** ** **");
	QN_OUTPUT("New epoch: epoch %i, learn rate %f.", epoch, learn_rate);
	QN_timestr(timebuf, sizeof(timebuf));
	QN_OUTPUT("Epoch started: %s.", timebuf);

	// Training phase.
	set_learnrate();	// Set the learning rate 
	train_ftr_str->rewind();
	train_targ_str->rewind();
	percent_correct = train_epoch();
	train_error = 100.0f - percent_correct;
	if (train_error < best_train_error)
	{
	    best_train_error = train_error;
	    best_train_epoch = epoch;
	}
	QN_timestr(timebuf, sizeof(timebuf));
	if (verbose)
	    QN_OUTPUT("Training finished/CV started: %s.", timebuf);

	// Cross validation phase.
	cv_ftr_str->rewind();
	cv_targ_str->rewind();
	percent_correct = cv_epoch();
	cv_error = 100.0f - percent_correct;

	// Keeping track of how well we are doing.
	if ((strcmp(lr_sched->get_state(), "list") != 0) && (cv_error > last_cv_error))	//cz277 - list restore
	{
	    QN_OUTPUT("Weight log: Restoring previous weights from "
	    "`%s\'.", last_weightlog_filename);
	    QN_readwrite_weights(debug, dbgname, 
				 *mlp, last_weightlog_filename,
				 wfile_format, QN_READ);
	}
	else
	{
	    int ec;		// Error code.

	    ec = QN_logfile_template_map(wlog_template,
					 last_weightlog_filename, MAXPATHLEN,
					 epoch, pid);
	    if (ec!=QN_OK)
	    {
		clog.error("failed to build weight log file name from "
			   "template \'%s\'.", wlog_template);
	    }
	    QN_OUTPUT("Weight log: Saving weights to `%s\'.",
		      last_weightlog_filename);
	    QN_readwrite_weights(debug, dbgname, *mlp, last_weightlog_filename,
				 wfile_format, QN_WRITE);
	    last_cv_error = cv_error;
	    best_cv_error = cv_error;
	    best_cv_epoch = epoch;
	}

	// Epoch end status.
	QN_timestr(timebuf, sizeof(timebuf));
	if (verbose)
	    QN_OUTPUT("Epoch finished: %s.", timebuf);

	// On to next epoch.
	learn_rate = lr_sched->next_rate(cv_error);
	epoch++;
    }

    // Wind down log messages.
    QN_OUTPUT("** ** ** ** ** ** ** ** ** ** ** ** ** **");
    QN_OUTPUT("Best CV accuracy: %.2f%% correct in epoch %i.",
	    100.0 - best_cv_error, (int) best_cv_epoch);
    QN_OUTPUT("Best train accuracy: %.2f%% correct in epoch %i.",
	    100.0 - best_train_error, (int) best_train_epoch);
    run_stop_time = QN_time();
    QN_timestr(timebuf, sizeof(timebuf));
    QN_OUTPUT("Training run stop: %s.", timebuf);
    double total_time = run_stop_time - run_start_time;
    int hours = ((int) total_time) / 3600;
    int mins = (((int) total_time) / 60) % 60;
    int secs = ((int) total_time) % 60;
    QN_OUTPUT("Training time: %.2f secs (%i hours, %i mins, %i secs).",
	      total_time, hours, mins, secs);
    QN_OUTPUT("** ** ** ** ** ** ** ** ** ** ** ** ** **");
}

// Note - cross validation is less demanding on the facilities that
// must be provided by the stream.
double
QN_SoftSentTrainer::cv_epoch()
{
    size_t ftr_count;		// Count of feature frames read.
    size_t targ_count;		// Count of target frames read.
    size_t total_frames = 0;	// Total frames read this phase.
    size_t correct_frames = 0;	// Number of correct frames this phase.
    size_t current_segno;	// Current segment number.
    double start_secs;		// Exact time we started.
    double stop_secs;		// Exact time we stopped.
    size_t i;			// Local counter.

    current_segno = 0;
    ftr_count = 0;		// Pretend that previous read hit end of seg.
    start_secs = QN_time();
    
    // Iterate over all input segments.
    // Note that, at this stage, an input segment is _not_ typically a
    // sentence, but a buffer full of sentences (known as a "fill" in old
    // QuickNet code).
    while (1)
    {
	if (ftr_count<bunch_size) // Check if at end of segment.
	{
	    QN_SegID ftr_segid;	// Segment ID from input stream.
	    QN_SegID targ_segid; // Segment ID from target stream.

	    ftr_segid = cv_ftr_str->nextseg();
	    targ_segid = cv_targ_str->nextseg();
	    assert(ftr_segid==targ_segid);
	    if (ftr_segid==QN_SEGID_BAD)
		break;
	    else
	    {
		current_segno++;
	    }
	}

	// Get the data to pass on to the net.
	ftr_count = cv_ftr_str->read_ftrs(bunch_size, inp_buf);
	targ_count = cv_targ_str->read_ftrs(bunch_size, targ_buf);
	if (ftr_count!=targ_count)
	{
	    clog.error("Feature and label streams have different segment "
		       "lengths in cross validation.");
	}
	// Do the forward pass.
	mlp->forward(ftr_count, inp_buf, out_buf);
	
	// Analyze the output of the net.
	float* out_buf_ptr = out_buf; // Current output frame.
	float* targ_buf_ptr = targ_buf; // Current target vector.
	QNUInt32 net_label;		// Label chosen by the net.
	QNUInt32 targ_label;		// Label given by targets.
	for (i=0; i<ftr_count; i++)
	{
	    net_label = qn_imax_vf_u(mlp_outs, out_buf_ptr);
	    targ_label = qn_imax_vf_u(mlp_outs, targ_buf_ptr);
	    if (net_label==targ_label)
		correct_frames++;
	    out_buf_ptr += mlp_outs;
	    targ_buf_ptr += mlp_outs;
	}
	total_frames += ftr_count;
    }
    stop_secs = QN_time();
    double total_secs = stop_secs - start_secs;
    double percent = 100.0 * (double) correct_frames / (double) total_frames;

    QN_OUTPUT("CV speed: %.2f MCPS, %.1f presentations/sec.",
	      QN_secs_to_MCPS(stop_secs-start_secs, total_frames, *mlp),
	      (double) total_frames / total_secs);
    QN_OUTPUT("CV accuracy:  %lu right out of %lu, %.2f%% correct.",
	      (unsigned long) correct_frames, (unsigned long) total_frames,
	      percent);

    return percent; 
}

double
QN_SoftSentTrainer::train_epoch()
{
    size_t ftr_count;		// Count of feature frames read.
    size_t targ_count;		// Count of target frames read.
    size_t total_frames = 0;	// Total frames read this phase.
    size_t seg_frames = 0;      // Number of frames in this segment.
    size_t correct_frames = 0;	// Number of correct frames this phase.
    size_t total_segs;		// Number of segments in streams.
    size_t current_segno;	// Current segment number.
    time_t current_time;	// Current time.
    double start_secs;		// Exact time we started.
    double stop_secs;		// Exact time we stopped.
    double total_secs;		// Total time.
    double percent;
    size_t i;			// Local counter.

    assert(bunch_size!=0);

    total_segs = train_ftr_str->num_segs();
    current_segno = 0;
    ftr_count = 0;		// Pretend that previous read hit end of seg.
    start_secs = QN_time();
    
    // Iterate over all input segments.
    // Note that, at this stage, an input segment is _not_ typically a
    // sentence, but a buffer full of sentences (known as a "fill" in old
    // QuickNet code).
    while (1)
    {
	if (ftr_count<bunch_size) // Check if at end of segment.
	{
	    QN_SegID ftr_segid;	// Segment ID from input stream.
	    QN_SegID targ_segid; // Segment ID from target stream.

	    // update the learning rate, for those schedules that care
	    float nlearn_rate=lr_sched->trained_on_nsamples(seg_frames);
	    if (nlearn_rate != learn_rate) {
	      learn_rate=nlearn_rate;
	      set_learnrate();  // Pass the info along to the MLP
	      if (verbose) 
		QN_OUTPUT("learning rate set to %.6f after %d samples read",learn_rate,seg_frames);
	    }
	    seg_frames=0;

	    ftr_segid = train_ftr_str->nextseg();
	    targ_segid = train_targ_str->nextseg();
	    assert(ftr_segid==targ_segid);
	    if (ftr_segid==QN_SEGID_BAD)
		break;
	    else
	    {
		if (verbose)
		{
		    QN_OUTPUT("Starting chunk %lu of %lu containing "
			      "%lu frames...",
			      (unsigned long) (current_segno+1),
			      (unsigned long) total_segs,
			      train_ftr_str->num_frames(current_segno));
		}
		current_segno++;
	    }
	}

	// Get the data to pass on to the net.
	ftr_count = train_ftr_str->read_ftrs(bunch_size, inp_buf);
	targ_count = train_targ_str->read_ftrs(bunch_size, targ_buf);
	if (ftr_count!=targ_count)
	{
	    clog.error("Feature and label streams have different "
		       "lengths in cross validation.");
	}

	// Do the training.
	mlp->train(ftr_count, inp_buf, targ_buf, out_buf);

	// Analyze the output of the net.
	float* out_buf_ptr = out_buf; // Current output frame.
	float* targ_buf_ptr = targ_buf; // Current target vector.
	QNUInt32 net_label;		// Label chosen by the net.
	QNUInt32 targ_label;		// Label from file.
	for (i=0; i<ftr_count; i++)
	{
	    net_label = qn_imax_vf_u(mlp_outs, out_buf_ptr);
	    targ_label = qn_imax_vf_u(mlp_outs, targ_buf_ptr);
	    if (net_label==targ_label)
		correct_frames++;
	    out_buf_ptr += mlp_outs;
	    targ_buf_ptr += mlp_outs;
	}
	total_frames += ftr_count;
	seg_frames += ftr_count;

	current_time = time(NULL);
	if (ckpt_secs!=0 && (current_time > (last_ckpt_time + ckpt_secs)))
	{
	    last_ckpt_time = current_time;
	    checkpoint_weights();
	}
    }
    stop_secs = QN_time();
    total_secs = stop_secs - start_secs;
    percent = 100.0 * (double) correct_frames / (double) total_frames;

    QN_OUTPUT("Train speed: %.2f MCUPS, %.1f presentations/sec.",
	      QN_secs_to_MCPS(stop_secs-start_secs, total_frames, *mlp),
	      (double) total_frames / total_secs);
    QN_OUTPUT("Train accuracy:  %lu right out of %lu, %.2f%% correct.",
	      (unsigned long) correct_frames, (unsigned long) total_frames,
	      percent);

    return percent; 
}

// Set the learning rate for the net according to the learn_rate variable.
void
QN_SoftSentTrainer::set_learnrate()
{
    size_t i;
    float scale;		// Learnrate scaler.

    for (i=0; i<mlp->num_sections(); i++)
    {
	scale = lrscale[i/2];
	mlp->set_learnrate((QN_SectionSelector) i, learn_rate*scale);
    }
}



void
QN_SoftSentTrainer::checkpoint_weights()
{
    int ec;
    char ckpt_filename[MAXPATHLEN]; // Checkpoint filename.
    
    ec = QN_logfile_template_map(ckpt_template,
				 ckpt_filename, MAXPATHLEN,
				 epoch, pid);
    if (ec!=QN_OK)
    {
	clog.error("failed to build ckpt weight file name from "
		   "template \'%s\'.", ckpt_template);
    }
    QN_OUTPUT("Checkpoint: Saving weights to `%s\'.",
	      ckpt_filename);
    QN_readwrite_weights(debug,dbgname, *mlp,
			 ckpt_filename, ckpt_format, QN_WRITE);

}



