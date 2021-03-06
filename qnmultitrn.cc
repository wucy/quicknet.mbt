#ifndef NO_RCSID
const char* qnmultitrn_rcsid = 
    "$Header: /u/drspeech/repos/quicknet2/qnmultitrn.cc,v 1.18 2011/09/06 22:12:05 davidj Exp $";
#endif

#include <QN_config.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef QN_HAVE_LIMITS_H
#include <limits.h>
#endif
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS (0)
#define EXIT_FAILURE (1)
#endif
#include <sys/types.h>
#ifdef QN_HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#ifdef QN_HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
#include <unistd.h>



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


//cw564 - mbt
map<string, float *> gen_spkwgt(const char * fn)
{
    map<string, float *> ret;
    ifstream ifs(fn);
    int n, dim;
    ifs >> n >> dim;
    for (int i = 0; i < n; ++ i)
    {
        char name[1000];
        ifs >> name;
        float * vect = new float[100];
        for (int j = 0; j < dim; ++ j)
        {
            ifs >> vect[j];
        }
        ret[name] = vect;
    }
    ifs.close();
    return ret;
}

map<int, string> gen_seg2spk(const char * fn)
{
    map<int ,string> ret;
    int n;
    ifstream ifs(fn);
    ifs >> n;
    for (int i = 0; i < n; ++ i)
    {
        int segid;
        char buf[1000];
        ifs >> segid >> buf;
        ret[segid] = buf;
    }
    ifs.close();
    return ret;
}



#if !QN_HAVE_DECL_SRAND48
extern "C" {
void srand48(long);
}
#endif

#ifdef QN_HAVE_SET_NEW_HANDLER
extern "C" {
    typedef void (*new_handler)(void);
    new_handler set_new_handler (new_handler);
}
#endif


#ifndef FILENAME_MAX
#define FILENAME_MAX (MAXPATHLEN)
#endif

#include "QuickNet.h"

static struct {
    const char* ftr1_file;
    const char* ftr1_format;
    int ftr1_width;
    const char* ftr2_file;
    const char* ftr2_format;
    int ftr2_width;
    const char* unary_file;
    const char* hardtarget_file;
    const char* hardtarget_format;
    const char* softtarget_file;
    const char* softtarget_format;
    int softtarget_width;
    const char* ftr1_norm_file;
    const char* ftr2_norm_file;
    int ftr1_ftr_start;
    int ftr2_ftr_start;
    int ftr1_ftr_count;
    int ftr2_ftr_count;
    int hardtarget_lastlab_reject;
    int window_extent;
    int ftr1_window_offset;
    int ftr2_window_offset;
    int unary_window_offset;
    int hardtarget_window_offset;
    int softtarget_window_offset;
    int ftr1_window_len;
    int ftr2_window_len;
    int ftr1_delta_order;
    int ftr1_delta_win;
    const char* ftr1_norm_mode_str;
    int ftr1_norm_mode;
    double ftr1_norm_am;
    double ftr1_norm_av;
    int ftr2_delta_order;
    int ftr2_delta_win;
    const char* ftr2_norm_mode_str;
    int ftr2_norm_mode;
    double ftr2_norm_am;
    double ftr2_norm_av;
    long train_cache_frames;
    int train_cache_seed;
    const char* train_sent_range;
    const char* cv_sent_range;
    QN_Arg_ListFloat init_random_bias_min;
    QN_Arg_ListFloat init_random_bias_max;
    QN_Arg_ListFloat init_random_weight_min;
    QN_Arg_ListFloat init_random_weight_max;
    int init_random_seed;
    const char* init_weight_file;
    const char* init_weight_format;
    const char* log_weight_file;
    const char* log_weight_format;
    const char* ckpt_weight_file;
    const char* ckpt_weight_format;
    int ckpt_hours;
    const char* out_weight_file;
    const char* out_weight_format;
    const char* learnrate_schedule;
    QN_Arg_ListFloat learnrate_vals;
    long learnrate_epochs;
    float learnrate_scale;
    int unary_size;
    QN_Arg_ListInt mlp_size;
    const char* mlp_output_type;
    QN_Arg_ListFloat mlp_lrmultiplier;
    int mlp_bunch_size;
    int use_cuda;
    int use_blas;
    int use_pp;
    int use_fe;
    int mlp_threads;
    const char* log_file;		// Stream for storing status messages.
    int verbose;
    int debug;			// Debug level.
    //cz277 - newbob
    float a_min_derror_ramp_start;
    float a_min_derror_stop;
    int a_min_iter_num;
    //cz277 - momentum
    float a_alpha_momentum;
    //cz277 - nonlinearity
    QN_Arg_ListStr mlp_hidden_type;	//cz277 - mul actv
    //cz277 - critera
    const char* backprop_criterion;
    //cz277 - rand init
    const char* random_init_type;
    //cz277 - device select
    int device_no;
    //cz277 - outmap
    const char *map_file_path;
    const char *map_opt;
    //cz277 - learn rate criterion
    const char *lr_ctr;
    //cz277 - nn fea bp
    int bp_num_layer;
    //cz277 - env var
    const char *env_var4dev_id;
    //cz277 - weight decay
    float weight_decay_factor;


    //cw564 - mbt -- number of basis
    int mbt_num_basis;
    //cw564 - mbt -- init spkr lambda file
    const char * mbt_init_lambda_file;
    //cw564 - mbt -- prefix of out lambda file (per iter)
    const char * mbt_out_lambda_file_prefix;
    //cw564 - mbt -- segid to spkrid map file
    const char * mbt_seg2spkr_file;

} config;

static void
set_defaults(void)
{
    static float default_learnrate[1] = { 0.008 };
    static float default_bias_min[2] = { -0.1,-4.1 };
    static float default_bias_max[2] = { 0.1,-3.9 };
    static float default_weight_min[1] = { -0.1 };
    static float default_weight_max[1] = { 0.1 };

    static float default_lrmultiplier[1] = { 1.0 };

    static int default_mlp_size[3] = { 153, 200, 56 };
    //cz277 - mul actv
    static char *default_mlp_hidden_type[1] = { "sigmoid" };
    
    config.ftr1_file = "";
    config.ftr1_format = "pfile";
    config.ftr1_width = 0;
    config.ftr2_file = "";
    config.ftr2_format = "pfile";
    config.ftr2_width = 0;
    config.unary_file = "";
    config.hardtarget_file = "";
    config.hardtarget_format = "";
    config.softtarget_file = "";
    config.softtarget_format = "pfile";
    config.softtarget_width = 0;
    config.ftr1_norm_file = "";
    config.ftr2_norm_file = "";
    config.ftr1_ftr_start = 0;
    config.ftr2_ftr_start = 0;
    config.ftr1_ftr_count = 0;
    config.ftr2_ftr_count = 0;
    config.hardtarget_lastlab_reject = 0;
    config.window_extent = 9;
    config.ftr1_window_offset = 0;
    config.ftr2_window_offset = 4;
    config.unary_window_offset = 3;
    config.hardtarget_window_offset = 0;
    config.softtarget_window_offset = 0;
    config.ftr1_window_len = 9;
    config.ftr2_window_len = 0;
    config.ftr1_delta_order = 0;
    config.ftr1_delta_win = 9;
    config.ftr1_norm_mode_str = NULL;
    config.ftr1_norm_mode = QN_NORM_FILE;
    config.ftr1_norm_am = QN_DFLT_NORM_AM;
    config.ftr1_norm_av = QN_DFLT_NORM_AV;
    config.ftr2_delta_order = 0;
    config.ftr2_delta_win = 9;
    config.ftr2_norm_mode_str = NULL;
    config.ftr2_norm_mode = QN_NORM_FILE;
    config.ftr2_norm_am = QN_DFLT_NORM_AM;
    config.ftr2_norm_av = QN_DFLT_NORM_AV;
    config.train_cache_frames = 100000;
    config.train_cache_seed = 0;
    config.train_sent_range = 0;
    config.cv_sent_range = "";
    config.init_random_bias_min.count = 2;
    config.init_random_bias_min.vals = &default_bias_min[0];
    config.init_random_bias_max.count = 2;
    config.init_random_bias_max.vals = &default_bias_max[0];

    config.init_random_weight_min.count = 1;
    config.init_random_weight_min.vals = &default_weight_min[0];
    config.init_random_weight_max.count = 1;
    config.init_random_weight_max.vals = &default_weight_max[0];

    config.init_random_seed = 0;
    config.init_weight_file = "";
    config.init_weight_format = "matlab";
    config.log_weight_file = "log%p.weights";
    config.log_weight_format = "matlab";
    config.ckpt_weight_file = "ckpt-%h-%t.weights";
    config.ckpt_weight_format = "matlab";
    config.ckpt_hours = 0;
    config.out_weight_file = "out.weights";
    config.out_weight_format = "matlab";
    config.learnrate_schedule = "newbob";
    config.learnrate_vals.count = 1;
    config.learnrate_vals.vals = &default_learnrate[0];
    config.learnrate_epochs = 9999;
    config.learnrate_scale = 0.5;
    config.unary_size = 0;
    config.mlp_size.count = 3;
    config.mlp_size.vals = &default_mlp_size[0];
    config.mlp_lrmultiplier.count = 1;
    config.mlp_lrmultiplier.vals = &default_lrmultiplier[0];
    config.mlp_output_type = "softmax";
    config.mlp_bunch_size = 16;
#ifdef QN_HAVE_LIBBLAS
    config.use_blas = 1;
#else
    config.use_blas = 0;
#endif
    config.use_pp = 1;
    config.use_fe = 0;
    config.use_cuda = 0;
    config.mlp_threads = 1;
    config.log_file = "-";
    config.verbose = 0;
    config.debug = 0;
    //cz277 - newbob
    config.a_min_derror_ramp_start = 0.5;
    config.a_min_derror_stop = 0.05;
    config.a_min_iter_num = -1;
    //cz277 - momentum
    config.a_alpha_momentum = 0.0;
    //cz277 - nonlinearity, mul actv
    config.mlp_hidden_type.count = 1;
    config.mlp_hidden_type.vals = &default_mlp_hidden_type[0]; 
    //cz277 - criteria
    config.backprop_criterion = "xentropy";
    //cz277 - rand init
    config.random_init_type = "normal";
    //cz277 - device select
    config.device_no = 0;
    //cz277 - outmap
    config.map_file_path = "";
    config.map_opt = "";
    //cz277 - learn rate criterion
    config.lr_ctr = "cv";
    //cz277 - nn fea bp
    config.bp_num_layer = -1;
    //cz277 - env var
    config.env_var4dev_id = "X_SGE_CUDA_DEVICE";
    //cz277 - weight decay
    config.weight_decay_factor = 0.0;

    //cw564 - mbt -- number of basis
    config.mbt_num_basis = 1;
    //cw564 - mbt -- init spkr lambda file
    config.mbt_init_lambda_file = "";
    //cw564 - mbt -- prefix of out lambda file (per iter)
    config.mbt_out_lambda_file_prefix = "";
    //cw564 - mbt -- segid to spkrid map file
    config.mbt_seg2spkr_file = "";
}

QN_ArgEntry argtab[] =
{
{ NULL, "QuickNet MLP training program version " QN_VERSION, QN_ARG_DESC },
{ "ftr1_file", "Input feature file", QN_ARG_STR,
  &(config.ftr1_file), QN_ARG_REQ },
{ "ftr1_format", "Main feature file format [pfile,pre,lna,onlftr,srifile,srilist,htk]", QN_ARG_STR,
  &(config.ftr1_format) },
{ "ftr1_width", "Main feature file feature columns", QN_ARG_INT,
  &(config.ftr1_width) },
{ "ftr2_file", "Second input feature file", QN_ARG_STR,
  &(config.ftr2_file) },
{ "ftr2_format","Secondary feature file format [pfile,pre,lna,onlftr,srifile,srilist,htk]", QN_ARG_STR,
  &(config.ftr2_format) },
{ "ftr2_width", "Secondary feature file feature columns", QN_ARG_INT,
  &(config.ftr2_width) },
{ "unary_file", "Auxilliary unary file", QN_ARG_STR,
  &(config.unary_file) },
{ "hardtarget_file", "Target label file", QN_ARG_STR,
  &(config.hardtarget_file) },
{ "hardtarget_format", "Target label file format [pfile,pre,ilab]", QN_ARG_STR,
  &(config.hardtarget_format) },
{ "softtarget_file", "Target feature file", QN_ARG_STR,
  &(config.softtarget_file) },
{ "softtarget_format", "Target feature file format [pfile,pre,lna,onlftr]", QN_ARG_STR,
  &(config.softtarget_format) },
{ "softtarget_width", "Target feature file feature columns", QN_ARG_INT,
  &(config.softtarget_width) },
{ "ftr1_norm_file", "Normalization parameters for ftr1_file", QN_ARG_STR,
  &(config.ftr1_norm_file) },
{ "ftr2_norm_file", "Normalization parameters for ftr2_file", QN_ARG_STR,
  &(config.ftr2_norm_file) },
{ "ftr1_ftr_start", "First feature used from ftr1_file",
  QN_ARG_INT, &(config.ftr1_ftr_start) },
{ "ftr2_ftr_start", "First feature used from ftr2_file",
  QN_ARG_INT, &(config.ftr2_ftr_start) },
{ "ftr1_ftr_count", "Number of features used from ftr1_file",
  QN_ARG_INT, &(config.ftr1_ftr_count) },
{ "ftr2_ftr_count", "Number of features used from ftr2_file",
  QN_ARG_INT, &(config.ftr2_ftr_count) },
{ "hardtarget_lastlab_reject", "Last label value indicates no-train frames",
  QN_ARG_BOOL, &(config.hardtarget_lastlab_reject) },
{ "window_extent", "Extent of all windows (frames)", QN_ARG_INT,
  &(config.window_extent) },
{ "ftr1_window_offset", "Offset of window on ftr1_file (frames)",
  QN_ARG_INT, &(config.ftr1_window_offset) },
{ "ftr2_window_offset", "Offset of window on ftr2_file (frames)",
  QN_ARG_INT, &(config.ftr2_window_offset) },
{ "unary_window_offset", "Offset of window on unary_file (frames)",
  QN_ARG_INT, &(config.unary_window_offset) },
{ "hardtarget_window_offset", "Offset of window on target label file (frames)",
  QN_ARG_INT, &(config.hardtarget_window_offset) },
{ "softtarget_window_offset", "Offset of window on target feature file (frames)",
  QN_ARG_INT, &(config.softtarget_window_offset) },
{ "ftr1_window_len", "Length of window on ftr1_file (frames)", QN_ARG_INT,
  &(config.ftr1_window_len) },
{ "ftr2_window_len", "Length of window on ftr2_file (frames)", QN_ARG_INT,
  &(config.ftr2_window_len) },
{ "ftr1_delta_order", "Order of derivatives added to ftr1_file", QN_ARG_INT,
  &(config.ftr1_delta_order) },
{ "ftr1_delta_win", "Window size for ftr1_file delta-calculation", QN_ARG_INT,
  &(config.ftr1_delta_win) },
{ "ftr1_norm_mode", "Normalization mode (file/utts/online)", QN_ARG_STR,
  &(config.ftr1_norm_mode_str) },
{ "ftr1_norm_alpha_m", "Update constant for online norm means", QN_ARG_DOUBLE,
  &(config.ftr1_norm_am) },
{ "ftr1_norm_alpha_v", "Update constant for online norm vars", QN_ARG_DOUBLE,
  &(config.ftr1_norm_av) },
{ "ftr2_delta_order", "Order of derivatives added to ftr2_file", QN_ARG_INT,
  &(config.ftr2_delta_order) },
{ "ftr2_delta_win", "Window size for ftr2_file delta-calculation", QN_ARG_INT,
  &(config.ftr2_delta_win) },
{ "ftr2_norm_mode", "Normalization mode (file/utts/online)", QN_ARG_STR,
  &(config.ftr2_norm_mode_str) },
{ "ftr2_norm_alpha_m", "Update constant for online norm means", QN_ARG_DOUBLE,
  &(config.ftr2_norm_am) },
{ "ftr2_norm_alpha_v", "Update constant for online norm vars", QN_ARG_DOUBLE,
  &(config.ftr2_norm_av) },
{ "train_cache_frames", "Number of training frames in cache",
  QN_ARG_LONG, &(config.train_cache_frames) },
{ "train_cache_seed", "Training presentation randomization seed",
  QN_ARG_INT, &(config.train_cache_seed) },
{ "train_sent_range", "Training sentence indices in QN_Range(3) format",
  QN_ARG_STR, &(config.train_sent_range) },
{ "cv_sent_range", "Cross validation sentence indices in QN_Range(3) format",
  QN_ARG_STR, &(config.cv_sent_range) },
{ "init_random_bias_min", "Minimum random bias (per layer)", QN_ARG_LIST_FLOAT,
  &(config.init_random_bias_min) },
{ "init_random_bias_max", "Maximum random bias (per layer)", QN_ARG_LIST_FLOAT,
  &(config.init_random_bias_max) },
{ "init_random_weight_min", "Minimum random weight (per layer)", QN_ARG_LIST_FLOAT,
  &(config.init_random_weight_min) },
{ "init_random_weight_max", "Maximum random weight (per layer)", QN_ARG_LIST_FLOAT,
  &(config.init_random_weight_max) },
{ "init_random_seed", "Net initialization random number seed",
  QN_ARG_INT, &(config.init_random_seed) },
{ "init_weight_file", "Input weight file", QN_ARG_STR,
  &(config.init_weight_file) },
{ "init_weight_format", "Input weight file format", QN_ARG_STR,
  &(config.init_weight_format) },
{ "log_weight_file", "Log weight file", QN_ARG_STR,
  &(config.log_weight_file) },
{ "log_weight_format", "Log weight file format", QN_ARG_STR,
  &(config.log_weight_format) },
{ "ckpt_weight_file", "Checkpoint weight file", QN_ARG_STR,
  &(config.ckpt_weight_file) },
{ "ckpt_weight_format", "Checkpoint weight file format", QN_ARG_STR,
  &(config.ckpt_weight_format) },
{ "ckpt_hours", "Checkpoint interval (in hours)", QN_ARG_INT,
  &(config.ckpt_hours) },
{ "out_weight_file", "Output weight file", QN_ARG_STR,
  &(config.out_weight_file) },
{ "out_weight_format", "Output weight file format", QN_ARG_STR,
  &(config.out_weight_format) },
{ "learnrate_schedule", "LR schedule type [newbob,list,smoothdecay]",
      QN_ARG_STR, &(config.learnrate_schedule) },
{ "learnrate_vals", "Learning rates",
      QN_ARG_LIST_FLOAT, &(config.learnrate_vals) },
{ "learnrate_epochs", "Maximum number of epochs", QN_ARG_LONG,
  &(config.learnrate_epochs) },
{ "learnrate_scale", "Scale factor of successive learning rates", QN_ARG_FLOAT,
  &(config.learnrate_scale) },
{ "unary_size", "Number of unary inputs to net",
  QN_ARG_INT, &(config.unary_size)},
{ "mlp_size", "Size of MLP",
  QN_ARG_LIST_INT, &(config.mlp_size)},
{ "mlp_lrmultiplier", "MLP per-section learning rate scale value",
  QN_ARG_LIST_FLOAT, &(config.mlp_lrmultiplier)},
{ "mlp_output_type","Type of non-linearity in MLP output layer [sigmoid,softmax,tanh,softsign,linear]",	//cz277 - nonlinearity
  QN_ARG_STR, &(config.mlp_output_type) },
{ "mlp_bunch_size","Size of bunches used in MLP training",
  QN_ARG_INT, &(config.mlp_bunch_size) },
{ "use_blas","Use BLAS libraries",
  QN_ARG_BOOL, &(config.use_blas) },
{ "use_pp","Use internal high-performance libraries",
  QN_ARG_BOOL, &(config.use_pp) },
{ "use_fe","Use fast exponent approximation for sigmoid, softmax, tanh etc.",
  QN_ARG_BOOL, &(config.use_fe) },
{ "use_cuda","Use CUDA GPU hardware",
  QN_ARG_BOOL, &(config.use_cuda) },
{ "mlp_threads","Number of threads in MLP object",
  QN_ARG_INT, &(config.mlp_threads) },
{ "log_file", "File for status messages", QN_ARG_STR, &(config.log_file) },
{ "verbose", "Output extra status messages",
  QN_ARG_BOOL, &(config.verbose) },
{ "debug", "Level of internal diagnostic output",
  QN_ARG_INT, &(config.debug) },
//cz277 - newbob
{ "a_min_derror_ramp_start", "Threashold for newbob", QN_ARG_FLOAT,
  &(config.a_min_derror_ramp_start) },
{ "a_min_derror_stop", "Threashold for newbob", QN_ARG_FLOAT,
  &(config.a_min_derror_stop) },
{ "min_iter_num", "Minimum number of iteration to run before ending newbob", QN_ARG_INT,
  &(config.a_min_iter_num) },
//cz277 - momentum
{ "a_alpha_momentum", "Alpha for momentum", QN_ARG_FLOAT, &(config.a_alpha_momentum) },
//cz277 - nonlinearity, mul actv
{ "mlp_hidden_type", "The nonlinear type of the hidden units (per network or per hidden layer) [sigmoid,softmax,tanh,softsign,linear]", QN_ARG_LIST_STR, &(config.mlp_hidden_type) }, 
//cz277 - criteria
{ "backprop_criterion", "The training criterion [quadratic,xentropy]", QN_ARG_STR, &(config.backprop_criterion) },
//cz277 - rand init
{ "random_init_type", "The random initialization type [normal, normalized]", QN_ARG_STR, &(config.random_init_type) },
//cz277 - device select
{ "device_no", "The index of the GPU device in use", QN_ARG_INT, &(config.device_no) },
//cz277 - outmap
{ "map_file_path", "The path of the mapping file", QN_ARG_STR, &(config.map_file_path) },
//cz277 - outmap
{ "map_opt", "Target mapping option [, max, sum, avg]", QN_ARG_STR, &(config.map_opt) },
//cz277 - learn rate criterion
{ "lr_ctr", "Learn rate criterion [cv, mapped_cv]", QN_ARG_STR, &(config.lr_ctr) },
//cz277 - nn fea bp
{ "bp_num_layer", "The number of layers got trained in back propagation", QN_ARG_INT, &(config.bp_num_layer) },
//cz277 - env var
{ "env_var4dev_id", "The environment variable indicating the right GPU device to use, set by SGE", QN_ARG_STR, &(config.env_var4dev_id) },
//cz277 - weight decay
{ "weight_decay_factor", "The weight decay factor", QN_ARG_FLOAT, &(config.weight_decay_factor) },

//cw564 - mbt
{ "mbt_num_basis", "[MBT] number of basis", QN_ARG_INT, &(config.mbt_num_basis) },
{ "mbt_init_lambda_file", "[MBT] init lambda file", QN_ARG_STR, &(config.mbt_init_lambda_file) },
{ "mbt_out_lambda_file_prefix", "[MBT] prefix of out lambda file (per iter)", 
    QN_ARG_STR, &(config.mbt_out_lambda_file_prefix) },
{ "mbt_seg2spkr_file", "[MBT] segid to spkrid map file", QN_ARG_STR, &(config.mbt_seg2spkr_file) },

{ NULL, NULL, QN_ARG_NOMOREARGS }  
};

//cz277 - mul actv
QN_LayerType *hiddenlayer_types;

// QN_open_ftrstream, QN_open_ftrfile and QN_close_ftrfiles all moved to QN_utils.cc

// A function to create a train and cross validation stream for a given
// feature file.  Also handles opening multiple files if 
// stream comes from a sequence of files.

void
create_ftrstreams(int debug, const char* dbgname, const char* filename,
		  const char* format, size_t width,
		  FILE* normfile, size_t first_ftr, size_t num_ftrs,
		  const char* train_sent_range, 
		  const char* cv_sent_range, 
		  size_t window_extent, size_t window_offset,
		  size_t window_len, 
		  int delta_order, int delta_win,  
		  int norm_mode, double norm_am, double norm_av, 
		  size_t train_cache_frames, int train_cache_seed,
		  QN_InFtrStream** train_str_ptr, QN_InFtrStream** cv_str_ptr)
{
    QN_InFtrStream* ftr_str = NULL;	// Temporary stream holder.
    int index = 1; 			// training always requires indexed
    int buffer_frames = 500;

    ftr_str = QN_build_ftrstream(debug, dbgname, filename, format, 
			      width, index, normfile, 
			      first_ftr, num_ftrs, 
			      0, QN_ALL, // do utt selection ourselves
			      buffer_frames, 
			      delta_order, delta_win, 
			      norm_mode, norm_am, norm_av);

    // Create training and cross-validation streams.
    QN_InFtrStream_Cut* train_ftr_str = NULL;
    QN_InFtrStream_Cut2* cv_ftr_str = NULL;

    // Using range strings
    QN_InFtrStream_CutRange* fwd_ftr_str 
	= new QN_InFtrStream_CutRange(debug, dbgname, *ftr_str, 
				      train_sent_range, 
				      cv_sent_range);
    train_ftr_str = (QN_InFtrStream_Cut*)fwd_ftr_str;
    cv_ftr_str = new QN_InFtrStream_Cut2(*train_ftr_str);

    // Create training and CV windows.
    size_t bot_margin = window_extent - window_offset - window_len;
    QN_InFtrStream_RandWindow* train_winftr_str =
	new QN_InFtrStream_RandWindow(debug, dbgname,
				      *train_ftr_str, window_len,
				      window_offset, bot_margin,
				      train_cache_frames, train_cache_seed
	    );
    QN_InFtrStream_SeqWindow* cv_winftr_str =
	new QN_InFtrStream_SeqWindow(debug, dbgname,
				      *cv_ftr_str, window_len,
				      window_offset, bot_margin
				      );
    *train_str_ptr = train_winftr_str;
    *cv_str_ptr = cv_winftr_str;
}

// A function to create a train and cross validation stream for a given
// label file.

void
create_labstreams(int debug, const char* dbgname, FILE* hardtarget_file,
		  const char* format, size_t width,
		  const char* train_sent_range, 
		  const char* cv_sent_range, 
		  size_t window_extent, size_t window_offset,
		  size_t train_cache_frames, int train_cache_seed,
		  QN_InLabStream** train_str_ptr, QN_InLabStream** cv_str_ptr)
{
    QN_InLabStream* lab_str;	// Temporary stream holder.

    // Convert the file descriptor into a stream.
    if (strcmp(format, "pfile")==0)
    {
	QN_InFtrLabStream_PFile* pfile_str =
	    new QN_InFtrLabStream_PFile(debug, // Select debugging.
					dbgname, // Debugging tag.
					hardtarget_file, // Label file.
					1 // Indexed flag.
		);
	if (pfile_str->num_labs()!=1)
	{
	    QN_ERROR("create_labstreams",
		     "Label file has %lu features, should only be 1.",
		     (unsigned long) pfile_str->num_labs() );
	}
	lab_str = pfile_str;
    }
    else if (strcmp(format, "pre")==0)
    {
	QN_InFtrLabStream_PreFile* prefile_str =
	    new QN_InFtrLabStream_PreFile(debug, // Select debugging.
					  dbgname, // Debugging tag.
					  hardtarget_file, // Label file.
					  width, // No of ftrs.
					  1 // Indexed flag.
		);
	lab_str = prefile_str;
    }
    else if (strcmp(format, "ilab")==0)
    {
	QN_InLabStream_ILab* ilab_str =
	    new QN_InLabStream_ILab(debug, // Select debugging.
				    dbgname, // Debugging tag.
				    hardtarget_file, // Label file.
				    1 // Indexed flag.
				    );
	lab_str = ilab_str;
    }
    else
    {
	QN_ERROR(dbgname, "unknown label file format '%s'.", format);
	lab_str = NULL;
    }
	

    // Create training and cross-validation streams.
    QN_InLabStream_Cut* train_lab_str = NULL;
    QN_InLabStream_Cut2* cv_lab_str = NULL;


    // Using range strings
    QN_InLabStream_CutRange* fwd_lab_str 
	= new QN_InLabStream_CutRange(debug, dbgname, *lab_str, 
				      train_sent_range, 
				      cv_sent_range);
    train_lab_str = (QN_InLabStream_Cut*)fwd_lab_str;
    cv_lab_str = new QN_InLabStream_Cut2(*train_lab_str);

    // Create training and CV windows.

    const size_t window_len = 1;
    size_t bot_margin = window_extent - window_offset - window_len;
    QN_InLabStream_RandWindow* train_winlab_str =
	new QN_InLabStream_RandWindow(debug, dbgname,
				      *train_lab_str, window_len,
				      window_offset, bot_margin,
				      train_cache_frames, train_cache_seed
	    );
    QN_InLabStream_SeqWindow* cv_winlab_str =
	new QN_InLabStream_SeqWindow(debug, dbgname,
				      *cv_lab_str, window_len,
				      window_offset, bot_margin
				      );
    *train_str_ptr = train_winlab_str;
    *cv_str_ptr = cv_winlab_str;
}

void
create_mlp(int debug, const char*,
	   size_t n_layers, size_t* layer_size,
	   const char* mlp_output_type, QN_Arg_ListStr mlp_hidden_type,	//cz277 - nonlinearity
	   int mlp_bunch_size, int threads, int cuda, int fe, 
	   QN_MLP** mlp_ptr, const char* backprop_criterion)	//cz277 - criteria
{
    // Create MLP and load weights.
    QN_MLP* mlp3 = NULL;

    QN_OutputLayerType outlayer_type;
    if (strcmp(mlp_output_type, "sigmoid")==0) {
	outlayer_type = QN_OUTPUT_SIGMOID;
    } else if (strcmp(mlp_output_type, "linear")==0) {	//cz277 - nonlinearity
	outlayer_type = QN_OUTPUT_LINEAR;
    } else if (strcmp(mlp_output_type, "softmax")==0) {
	outlayer_type = QN_OUTPUT_SOFTMAX;
    } else if (strcmp(mlp_output_type, "tanh")==0) {
	outlayer_type = QN_OUTPUT_TANH;
    } else if (strcmp(mlp_output_type, "softsign")==0) {	//cz277 - nonlinearity
	outlayer_type = QN_OUTPUT_SOFTSIGN;
    } else {
	QN_WARN("create_mlp", "unknown output unit type '%s'.", mlp_output_type);
	outlayer_type = QN_OUTPUT_SOFTMAX;	//cz277 - nonlinearity
    }

    //cz277 - nonlinearity, mul actv
    if (mlp_hidden_type.count != 1 && mlp_hidden_type.count < (n_layers - 2)) {
        QN_ERROR("create_mlp", "mlp_hidden_type segment number %d do not match the hidden layer number %d.", mlp_hidden_type.count, n_layers - 2);
    }
    hiddenlayer_types = new QN_LayerType[n_layers];
    hiddenlayer_types[0] = QN_LAYER_LINEAR;	//meaningless
    hiddenlayer_types[n_layers - 1] = QN_LAYER_LINEAR;	//meaningless
    for (int i = 0; i < n_layers - 2; ++i) {
	int validx = i;
	//the hidden layer types are specified in per network mode
	if (mlp_hidden_type.count == 1) {
	    validx = 0;
	}
        if (strcmp(mlp_hidden_type.vals[validx], "sigmoid")==0) {
	    hiddenlayer_types[i + 1] = QN_LAYER_SIGMOID; 
	    QN_OUTPUT("Layer %d, hidden activation type: sigmoid", i + 1);
        } else if (strcmp(mlp_hidden_type.vals[validx], "linear")==0) {
	    hiddenlayer_types[i + 1] = QN_LAYER_LINEAR;
	    QN_OUTPUT("Layer %d, hidden activation type: linear", i + 1);
        } else if (strcmp(mlp_hidden_type.vals[validx], "softmax")==0) {
    	    hiddenlayer_types[i + 1] = QN_LAYER_SOFTMAX;
	    QN_OUTPUT("Layer %d, hidden activation type: softmax", i + 1);
        } else if (strcmp(mlp_hidden_type.vals[validx], "tanh")==0) {
	    hiddenlayer_types[i + 1] = QN_LAYER_TANH;
	    QN_OUTPUT("Layer %d, hidden activation type: tanh", i + 1);
        } else if (strcmp(mlp_hidden_type.vals[validx], "softsign")==0) {
	    hiddenlayer_types[i + 1] = QN_LAYER_SOFTSIGN;
	    QN_OUTPUT("Layer %d, hidden activation type: softsign", i + 1);
        } else {
            QN_WARN("create_mlp", "unknown hidden out unit type '%s' at layer '%d'.", mlp_hidden_type, i + 1);
            hiddenlayer_types[i + 1] = QN_LAYER_SIGMOID;
        }

	//cz277 - debug
	/*printf("layer %d, dstidx %d, dstval = %s, addr = %x, newval = %d\n", i + 1, validx, mlp_hidden_type.vals[validx], hiddenlayer_types, hiddenlayer_types[i + 1]);*/
    }

    //cz277 - criteria
    QN_CriterionType criterion_type;
    if (strcmp(backprop_criterion, "quadratic")==0) {
	criterion_type = QN_CRITERION_QUADRATIC;
    } else if (strcmp(backprop_criterion, "xentropy")==0) {
	criterion_type = QN_CRITERION_XENTROPY;
    } else {
        QN_ERROR("create_mlp", "unknown training criterion '%s'.",
                 backprop_criterion);
	criterion_type = QN_CRITERION_XENTROPY;
    }

    if (mlp_bunch_size == 0) {
	QN_ERROR("create_mlp", "use of online nets not supported - "
		 "use a bunch size of 1 to get the same effect with "
		 "a bunch-mode net.");
    } else {
	// Bunch
	if (cuda)
	{
#ifdef QN_CUDA
	    if (threads>1)
	    {
		QN_WARN("create_mlp", "thread setting ignored on CUDA");
	    }
	    if (fe)
	    {
		QN_WARN("create_mlp", "use_fe setting ignored on CUDA");
	    }

	    QN_cuda_init(config.device_no, config.env_var4dev_id);	//cz277 - device select, env var
	    char devstr[128];
	    sprintf(devstr, "CUDA using device: %i.", QN_cuda_current_device());
	    QN_OUTPUT(devstr);

	    mlp3 = new QN_MLP_BunchCudaVar(config.bp_num_layer,	//cz277 - nn fea bp
					   debug, "train",
					   n_layers, layer_size,
					   criterion_type, hiddenlayer_types,	//cz277 - criteria	//cz277 - nonlinearity, mul actv
					   outlayer_type, mlp_bunch_size, config.device_no, config.env_var4dev_id);	//cz277 - device select, env var
	    QN_OUTPUT("MLP type: QN_MLP_BunchCudaVar.");
#else
	    QN_ERROR(NULL, "no CUDA support included with this build");
#endif
	}
	else 
	{
            // Not cuda
	    if (threads>1)
	    {
#ifdef QN_HAVE_LIBPTHREAD
		if (threads>mlp_bunch_size)
		{
		    QN_ERROR("create_mlp", "number of threads must "
			     "be less than the bunch size.");
		}
		else
		{
		    // Bunch threaded
		    mlp3 = new QN_MLP_ThreadFlVar(config.bp_num_layer,	//cz277 - nn fea bp
						  debug, "train",
						  n_layers, layer_size,
                                                  criterion_type, hiddenlayer_types,    //cz277 - criteria      //cz277 - nonlinearity, mul actv
						  outlayer_type, mlp_bunch_size, threads);
		    QN_OUTPUT("MLP type: QN_MLP_ThreadFlVar.");
		}
#else
		QN_ERROR("create_mlp",
			 "cannot use multiple threads as libpthread "
			 "was not linked with this executable.");
#endif
	    }
	    else if (threads==1)
	    {
		mlp3 = new QN_MLP_BunchFlVar(config.bp_num_layer,	//cz277 - nn fea bp
					     debug, "train",
					     n_layers, layer_size,
					     criterion_type, hiddenlayer_types,    //cz277 - criteria      //cz277 - nonlinearity, mul actv
					     outlayer_type, mlp_bunch_size);
		QN_OUTPUT("MLP type: QN_MLP_BunchFlVar.");
	    }
	    else
	    {
		    QN_ERROR("create_mlp","threads must be >= 1.");
	    }
	}
    }
    
    //cz277 - momentum
    ((QN_MLP_BaseFl *)mlp3)->set_alpha_momentum(config.a_alpha_momentum);
    //cz277 - weight decay
    ((QN_MLP_BaseFl *)mlp3)->set_weight_decay_factor(config.weight_decay_factor);
    
    *mlp_ptr = mlp3;
}

void
create_learnrate_schedule(int, const char*,
			  const char* learnrate_schedule,
			  float* learnrate_vals,
			  size_t learnrate_count,
			  float learnrate_scale,
			  size_t learnrate_epochs,
			  QN_RateSchedule** lr_schedule)
{
    QN_RateSchedule* rate_sched;
    if (learnrate_scale>1.0)
    {
	QN_ERROR("create_learnrate_schedule", "Learning rate scale is %g, but "
		 "it should be less that 1.0.");
    } 
    if (strcmp(learnrate_schedule, "newbob")==0)
    {
	//cz277 - newbob
	rate_sched = new QN_RateSchedule_NewBoB(*learnrate_vals,
						learnrate_scale,
						config.a_min_derror_ramp_start, config.a_min_derror_stop,
						100.0f,learnrate_epochs, config.a_min_iter_num);
    }
    else if (strcmp(learnrate_schedule, "list")==0)
    {
	long count;

	if (learnrate_epochs < learnrate_count)
	    count = learnrate_epochs;
	else
	    count = learnrate_count;
	rate_sched = new QN_RateSchedule_List(learnrate_vals, count);
    }
    else if (strcmp(learnrate_schedule, "smoothdecay")==0)
      {
	size_t search_epochs;

	if (learnrate_count<3 || learnrate_count>4) {
	  QN_ERROR(NULL,"learnrate_vals should have 3 or 4 values if learnrate_schedule is smoothdecay");
	}
	
	if (learnrate_count==4) {
	  search_epochs=(size_t)learnrate_vals[3];
	} else {
	  search_epochs=1;
	}

	QN_OUTPUT("Setting up smooth decay learning rate (lr=%.6f,decay=%.6f,stopcriterion=%.6f",learnrate_vals[0],learnrate_vals[1],learnrate_vals[2]);
	rate_sched = new QN_RateSchedule_SmoothDecay(learnrate_vals[0],
						     learnrate_vals[1],
						     learnrate_vals[2],
						     search_epochs,
						     100.0f, 0,
						     learnrate_epochs);
      }
    else
    {
	QN_ERROR("create_learnrate_schedule",
		 "Unknown learning rate schedule '%s'.",
		 learnrate_schedule);
	rate_sched = NULL;
    }
    *lr_schedule = rate_sched;
}

void
qnmultitrn()
{
    int verbose = config.verbose;
    time_t now;

    time(&now);

// A note for the logfile, including some system info.
    QN_output_sysinfo("qnmultitrn");
    QN_OUTPUT("Program start: %.24s.", ctime(&now));

// Open files and provisionally check arguments.
    if (verbose>0)
    {
	QN_OUTPUT("Opening feature file...");
    }


    // unary_file.
    enum { UNARYFILE_BUF_SIZE = 0x8000 };
    const char* unary_file = config.unary_file;
    FILE* unary_fp = NULL;
    if (strcmp(unary_file, "")!=0)
    {
	unary_fp = QN_open(unary_file, "r", UNARYFILE_BUF_SIZE, "unary_file");
    }

    const char* hardtarget_file = config.hardtarget_file;
    const char* softtarget_file = config.softtarget_file;
    FILE* hardtarget_fp = NULL;
    int lastlab_reject = config.hardtarget_lastlab_reject;
    if (strcmp(hardtarget_file, "")!=0 && strcmp(softtarget_file, "")==0)
    {
	// hardtarget_file.
	enum { LABFILE_BUF_SIZE = 0x8000 };
	hardtarget_fp = QN_open(hardtarget_file, "r", LABFILE_BUF_SIZE,
				"hardtarget_file");
    }
    else if (strcmp(hardtarget_file, "")==0 && strcmp(softtarget_file, "")!=0)
    {
	// opened within create_ftrstream
	if (lastlab_reject)
	{
	    QN_ERROR(NULL, "hardtarget_lastlab_reject cannot be true if no "
		     "hardtarget_file is specified");
	}
    }
    else
    {
	QN_ERROR(NULL, "must specify one and only one of hardtarget_file "
		 "and softtarget_file");
    }
	

    // ftr1_norm_file.
    FILE* ftr1_norm_fp = NULL;
    const char* ftr1_norm_file = config.ftr1_norm_file;
    if (strcmp(ftr1_norm_file, "")!=0)
    {
	ftr1_norm_fp = QN_open(ftr1_norm_file, "r");
    }
    
    // ftr2_norm_file.
    FILE* ftr2_norm_fp = NULL;
    const char* ftr2_norm_file = config.ftr2_norm_file;
    if (strcmp(ftr2_norm_file, "")!=0)
    {
	if (strcmp(config.ftr2_file, "")==0)
	    QN_ERROR(NULL, "ftr2_norm_file is specified but ftr2_file "
		     "is not.");
	else if (config.ftr2_ftr_count==0)
	    QN_ERROR(NULL, "ftr2_norm_file is specified but ftr2_ftr_count "
		     "is 0.");
	else
	    ftr2_norm_fp = QN_open(ftr2_norm_file, "r");
    }

    // Weight files.
    FILE* init_weight_fp = NULL;
    const char* init_weight_file = config.init_weight_file;
    if (strcmp(init_weight_file, "")!=0)
    {
	init_weight_fp = QN_open(init_weight_file, "r");
    }
    FILE* out_weight_fp = NULL;
    const char* out_weight_file = config.out_weight_file;
    out_weight_fp = QN_open(out_weight_file, "w");

    // Windowing.
    int window_extent = config.window_extent;
    if (window_extent<0 || window_extent>1000)
    {
	QN_ERROR(NULL, "window_extent must be in range 0-1000.");
    }
    int ftr1_window_offset = config.ftr1_window_offset;
    if (ftr1_window_offset<0 || ftr1_window_offset>=window_extent)
    {
	QN_ERROR(NULL, "ftr1_window_offset must be less than "
		 " window_extent.");
    }
    int ftr1_window_len = config.ftr1_window_len;
    if (ftr1_window_len<=0)
    {
	QN_ERROR(NULL, "ftr1_window_len must be greater than 0.");
    }
    if ((ftr1_window_offset + ftr1_window_len) > window_extent)
    {
	QN_ERROR(NULL, "ftr1_window_offset+ftr1_window_len must be "
		 "less than window_extent.");
    }
    int ftr2_window_offset = config.ftr2_window_offset;
    int ftr2_window_len = config.ftr2_window_len;
    // don't test ftr2_window_offset unless we have a file
    if (strcmp(config.ftr2_file, "")!= 0 && config.ftr2_ftr_count > 0) {
      if (ftr2_window_offset<0 || ftr2_window_offset>=window_extent)
        {
          QN_ERROR(NULL, "ftr2_window_offset must be less than "
                   " window_extent.");
        }
      if (ftr2_window_len<0)
        {
          QN_ERROR(NULL, "ftr2_window_len must be positive.");
        }
      if ((ftr2_window_offset + ftr2_window_len) > window_extent)
        {
          QN_ERROR(NULL, "ftr2_window_offset+ftr2_window_len must be "
                   "less than window_extent.");
        }
    }
    // Don't worry about the unary_window_offset unless there is actually
    // a unary_file (default value of 3 causes error for window_extent=1)
    int unary_window_offset = config.unary_window_offset;
    if ( (strcmp(unary_file, "")!=0) \
         && (unary_window_offset<0 || unary_window_offset>=window_extent))
    {
	QN_ERROR(NULL, "unary_window_offset must be less than "
		 " window_extent.");
    }
    int hardtarget_window_offset = config.hardtarget_window_offset;
    if (hardtarget_window_offset<0 || hardtarget_window_offset>=window_extent)
    {
	QN_ERROR(NULL, "hardtarget_window_offset must be less than "
		 " window_extent.");
    }
    int softtarget_window_offset = config.softtarget_window_offset;
    if (softtarget_window_offset<0 || softtarget_window_offset>=window_extent)
    {
	QN_ERROR(NULL, "softtarget_window_offset must be less than "
		 " window_extent.");
    }

    // Check for overlapping training and CV ranges.
    const char* train_sent_range = config.train_sent_range;
    const char* cv_sent_range = config.cv_sent_range;

    // Check for mlp_input_size consistency.
    size_t ftr1_ftr_start = config.ftr1_ftr_start;
    size_t ftr2_ftr_start = config.ftr2_ftr_start;
    size_t ftr1_ftr_count = config.ftr1_ftr_count;
    size_t ftr2_ftr_count = config.ftr2_ftr_count;
    size_t unary_size = config.unary_size;
    size_t ftrfile_num_input = ftr1_ftr_count  * ftr1_window_len
	+ ftr2_ftr_count * ftr2_window_len + unary_size;
    size_t mlp_layers = config.mlp_size.count;
    if (mlp_layers<2 || mlp_layers>QN_MLP_MAX_LAYERS)
    {
	QN_ERROR(NULL, "number of MLP layers must be between 2 and %lu.",
		 (unsigned long) QN_MLP_MAX_LAYERS);
    }
    size_t mlp_input_size = config.mlp_size.vals[0];
    size_t mlp_output_size = config.mlp_size.vals[mlp_layers-1];
    size_t mlp_layer_size[QN_MLP_MAX_LAYERS];
    size_t max_layer_size = 0;
    int bunch_size = config.mlp_bunch_size;
    size_t i;

    for (i=0; i<mlp_layers; i++)
    {
	int layer_size;

	layer_size = config.mlp_size.vals[i];
	if (layer_size<1)
	    QN_ERROR(NULL, "all MLP layer sizes must be >0.");
	if (layer_size>max_layer_size)
	    max_layer_size = layer_size;
	mlp_layer_size[i] = layer_size;
    }
#ifdef QN_CUDA
    if (config.use_cuda)
    {
	size_t maxvec = QN_cuda_maxvec();
	if ( (max_layer_size * bunch_size) > maxvec )
	{
	    QN_ERROR(NULL, "bunch size too big for CUDA library routings - max bunch size with this size net is %lu.", (unsigned long) maxvec / max_layer_size);
	}
    }
#endif


    //cw564 - mbt -- TODO
    /*if (ftrfile_num_input!=mlp_input_size)
    {
	QN_ERROR(NULL, "number of inputs to the net (%lu) does not equal width"
		 " of data stream from feature files (%lu).",
		 (unsigned long) mlp_input_size, 
		 (unsigned long) ftrfile_num_input);
    }*/
    

    // Sentence and randomization details.
    long train_cache_frames = config.train_cache_frames;
    int train_cache_seed = config.train_cache_seed;
    if (train_cache_frames<1000)
    {
	QN_ERROR(NULL, "train_cache_frames must be greater than 1000.");
    }

    
    int init_random_seed = config.init_random_seed;
    int debug = config.debug;

    // Do ftr1_file stream creation.
    QN_InFtrStream* ftr1_train_str = NULL;
    QN_InFtrStream* ftr1_cv_str = NULL;
    create_ftrstreams(debug, "ftr1_file", config.ftr1_file,
		      config.ftr1_format, config.ftr1_width,
		      ftr1_norm_fp,
		      ftr1_ftr_start, ftr1_ftr_count,
		      train_sent_range, 
		      cv_sent_range, 
		      window_extent,
		      ftr1_window_offset, ftr1_window_len,
		      config.ftr1_delta_order, config.ftr1_delta_win, 
		      config.ftr1_norm_mode, 
		      config.ftr1_norm_am, config.ftr1_norm_av,  
		      train_cache_frames, train_cache_seed,
		      &ftr1_train_str, &ftr1_cv_str);
		      
    // Do ftr2_file stream creation.
    QN_InFtrStream* ftr2_train_str = NULL;
    QN_InFtrStream* ftr2_cv_str = NULL;
    if (strcmp(config.ftr2_file, "")!=0)
    {
	if (config.ftr2_ftr_count==0)
	    QN_WARN(NULL, "ftr2_file is set but ftr2_ftr_count is 0.");
	create_ftrstreams(debug, "ftr2_file", config.ftr2_file,
			  config.ftr2_format, config.ftr2_width,
			  ftr2_norm_fp,
			  ftr2_ftr_start, ftr2_ftr_count,
			  train_sent_range, 
			  cv_sent_range, 
			  window_extent,
			  ftr2_window_offset, ftr2_window_len,
			  config.ftr2_delta_order, config.ftr2_delta_win, 
			  config.ftr2_norm_mode, 
			  config.ftr2_norm_am, config.ftr2_norm_av,  
			  train_cache_frames, train_cache_seed,
			  &ftr2_train_str, &ftr2_cv_str);
    }

    // Merge the two training feature streams.
    QN_InFtrStream* ftrfile_train_str;
    QN_InFtrStream* ftrfile_cv_str;
    if (ftr2_train_str!=NULL)
    {
	assert(ftr2_cv_str!=NULL);
	ftrfile_train_str = new QN_InFtrStream_JoinFtrs(debug, "train_ftrfile",
						       *ftr1_train_str,
 						       *ftr2_train_str);
	ftrfile_cv_str = new QN_InFtrStream_JoinFtrs(debug, "cv_ftrfile",
						     *ftr1_cv_str,
						     *ftr2_cv_str);
    }
    else
    {
	assert(ftr2_cv_str==NULL);
	assert(ftr2_train_str==NULL);
	ftrfile_train_str = ftr1_train_str;
	ftrfile_cv_str = ftr1_cv_str;
    }

    // If necessary, add the unary input feature.
    if (unary_fp!=NULL)
    {
	QN_InLabStream* unary_train_str = NULL;
	QN_InLabStream* unary_cv_str = NULL;
	
	create_labstreams(debug, "unary", unary_fp,
			  "pfile", 0,
			  train_sent_range, 
			  cv_sent_range, 
			  window_extent,
			  unary_window_offset,
			  train_cache_frames, train_cache_seed,
			  &unary_train_str, &unary_cv_str);

	// Convert the unary input label into a feature stream.
	QN_InFtrStream* unaryftr_train_str = NULL;
	QN_InFtrStream* unaryftr_cv_str = NULL;
	
	unaryftr_train_str = new QN_InFtrStream_OneHot(debug,
						       "train_unaryfile",
						       *unary_train_str,
						       unary_size);
	unaryftr_cv_str = new QN_InFtrStream_OneHot(debug,
						    "cv_unaryfile",
						    *unary_cv_str,
						    unary_size);
						   
	// Merge in the feature streams.
	ftrfile_train_str = new QN_InFtrStream_JoinFtrs(debug,
							"train_unaryfile",
							*ftrfile_train_str,
							*unaryftr_train_str);
	ftrfile_cv_str = new QN_InFtrStream_JoinFtrs(debug, "cv_unaryfile",
						     *ftrfile_cv_str,
						     *unaryftr_cv_str);
	
    }
    

    QN_InLabStream* hardtarget_train_str = NULL;
    QN_InLabStream* hardtarget_cv_str = NULL;
    QN_InFtrStream* softtarget_train_str = NULL;
    QN_InFtrStream* softtarget_cv_str = NULL;

    // Does config.ftr1_file refer to just a single file?
    int ftr1_onefile = 1;
    if (strchr(config.ftr1_file, ',') != NULL) {
	// filename looks like a comma-separated list
	ftr1_onefile = 0;
	// won't try to run pathcmp on it.
    }

    if (hardtarget_fp!=NULL)
    {
	// Do hardtarget stream creation.

	// Handle formats where we need to know the number of ftrs to
	// extract the labels.
	// A bit of a hack!!
	size_t hardtarget_width;
	if (ftr1_onefile && QN_pathcmp(config.ftr1_file, hardtarget_file)==0)
	    hardtarget_width = config.ftr1_width;
	else
	    hardtarget_width = 0;
	const char* hardtarget_format = config.hardtarget_format;
	if (strcmp(hardtarget_format, "")==0)
	    hardtarget_format = config.ftr1_format;
	
	create_labstreams(debug, "hardtarget", hardtarget_fp,
			  hardtarget_format, hardtarget_width,
			  train_sent_range, 
			  cv_sent_range, 
			  window_extent,
			  hardtarget_window_offset,
			  train_cache_frames, train_cache_seed,
			  &hardtarget_train_str, &hardtarget_cv_str);
    }
    else if (strcmp(softtarget_file,"")!=0)
    {
	size_t softtarget_width = config.softtarget_width;
	const char* softtarget_format = config.softtarget_format;
	if (strcmp(softtarget_format, "")==0)
	    softtarget_format = config.ftr1_format;
	
	create_ftrstreams(debug, "softtarget", softtarget_file,
			  softtarget_format, softtarget_width,
			  NULL,
			  0, 0,
			  train_sent_range, 
			  cv_sent_range, 
			  window_extent,
			  softtarget_window_offset, 1,
			  0, 0, 0,  /* no deltas or per-utt normalization */
			  0.0, 0.0, 
			  train_cache_frames, train_cache_seed,
			  &softtarget_train_str, &softtarget_cv_str);
	
    }
    else
	assert(0);
   

    // Create the MLP.
    QN_output_mlp_size("", mlp_layers, mlp_layer_size);
    QN_MLP* mlp;
    
    create_mlp(debug, "mlp", mlp_layers, mlp_layer_size,
	       config.mlp_output_type, config.mlp_hidden_type,	//cz277 - nonlinearity
	       config.mlp_bunch_size, config.mlp_threads, config.use_cuda,
	       config.use_fe, &mlp, config.backprop_criterion);	//cz277 - criteria
    
    mlp->mbt_num_basis = config.mbt_num_basis; //cw564 - mbt


    // Create the leaning rate schedule.
    QN_RateSchedule* lr_schedule;
    create_learnrate_schedule(debug, "learnrate",
			      config.learnrate_schedule,
			      config.learnrate_vals.vals,
			      config.learnrate_vals.count,
			      config.learnrate_scale,
			      config.learnrate_epochs,
			      &lr_schedule);

    

    const size_t num_sections = (mlp_layers-1)*2;
    const size_t num_subsections = mlp_layers - 1;

    // Handle per-section scaling of learing rates.
    float lrmultipliers[QN_MLP_MAX_LAYERS-1];
    int lrmultiplier_includes_zero = 0;
    if (config.mlp_lrmultiplier.count==1)
    {
	qn_copy_f_vf(num_subsections, config.mlp_lrmultiplier.vals[0], lrmultipliers);
	lrmultiplier_includes_zero = (config.mlp_lrmultiplier.vals[0]==0.0);
    }
    else
    {
	if (config.mlp_lrmultiplier.count!=num_subsections)
	    QN_ERROR(NULL, "number of mlp_lrmultiplier values must be 1 or %lu.",
		     (unsigned long) num_subsections);
	else
	{
	    for (i=0; i<num_subsections; i++)
	    {
		lrmultipliers[i] = config.mlp_lrmultiplier.vals[i];
		lrmultiplier_includes_zero |= (config.mlp_lrmultiplier.vals[i]==0.0);
	    }
	}
    }
    if (lrmultiplier_includes_zero)
    {
	QN_WARN(NULL,
		"mlp_lrmultiplier includes zeros, MCUPs number may be inaccurate.");
    }
	

    // A weight file of "" means randomize.
    if (init_weight_fp==NULL)
    {
	if (verbose>0)
	{
	    QN_OUTPUT("Randomizing weights...");
	}

	// If we have one limit, do all weights/biases to this.
	// If there are two limits, do output different from rest.
	float bias_mins[QN_MLP_MAX_LAYERS-1];
	float bias_maxs[QN_MLP_MAX_LAYERS-1];
	float weight_mins[QN_MLP_MAX_LAYERS-1];
	float weight_maxs[QN_MLP_MAX_LAYERS-1];

	// Sort out random weight initialization.
	// Note 1, 2 and n initializers all handled differently.
	if (config.init_random_weight_min.count==1 &&
	    config.init_random_weight_max.count==1)
	{
	    qn_copy_f_vf(num_subsections,
			 config.init_random_weight_min.vals[0],
			 weight_mins);
	    qn_copy_f_vf(num_subsections,
			 config.init_random_weight_max.vals[0],
			 weight_maxs);
	}
	else if (config.init_random_weight_min.count==2 &&
	    config.init_random_weight_max.count==2)
	{
	    if (num_subsections<2)
		QN_ERROR(NULL,"Can't have two random weight init values "
			 "with a two layer net.");
	    else
	    {
		for (i=0; i<num_subsections-1;i++)
		{
		    weight_mins[i] = config.init_random_weight_min.vals[0];
		    weight_maxs[i] = config.init_random_weight_max.vals[0];
		}
		weight_mins[num_subsections-1] =
		    config.init_random_weight_min.vals[1];
		weight_maxs[num_subsections-1] =
		    config.init_random_weight_max.vals[1];
	    }
	}
	else if (config.init_random_weight_min.count==num_subsections &&
	    config.init_random_weight_max.count==num_subsections)
	{
	    for (i=0; i<num_subsections;i++)
	    {
		weight_mins[i] = config.init_random_weight_min.vals[i];
		weight_maxs[i] = config.init_random_weight_max.vals[i];
	    }
	}
	else
	    QN_ERROR(NULL, "invalid number of init_random_weight_{min,max} "
		     "values - must be 1, 2 or %lu.",
		     (unsigned long) num_subsections);

	if (config.init_random_bias_min.count==1 &&
	    config.init_random_bias_max.count==1)
	{
	    for (i=0; i<num_subsections; i++)
	    {
		bias_mins[i] = config.init_random_bias_min.vals[0];
		bias_maxs[i] = config.init_random_bias_max.vals[0];
	    }
	}
	else if (config.init_random_bias_min.count==2 &&
	    config.init_random_bias_max.count==2)
	{
	    if (num_subsections<2)
		QN_ERROR(NULL,"Can't have two random bias init values "
			 "with a two layer net.");
	    else
	    {
		for (i=0; i<num_subsections-1;i++)
		{
		    bias_mins[i] = config.init_random_bias_min.vals[0];
		    bias_maxs[i] = config.init_random_bias_max.vals[0];
		}
		bias_mins[num_subsections-1] =
		    config.init_random_bias_min.vals[1];
		bias_maxs[num_subsections-1] =
		    config.init_random_bias_max.vals[1];
	    }
	}
	else if (config.init_random_bias_min.count==num_subsections &&
	    config.init_random_bias_max.count==num_subsections)
	{
	    for (i=0; i<num_subsections;i++)
	    {
		bias_mins[i] = config.init_random_bias_min.vals[i];
		bias_maxs[i] = config.init_random_bias_max.vals[i];
	    }
	}
	else
	    QN_ERROR(NULL, "invalid number of int_random_bias_{min,max} "
		     "values, must be 1, 2 or %lu.",
		     (unsigned long) num_subsections);
	QN_randomize_weights(debug, init_random_seed, *mlp,
			     weight_mins, weight_maxs,
			     bias_mins, bias_maxs);

	if (verbose>0)
	{
	    QN_OUTPUT("Randomized weights.");
	}
    }
    else
    {
	QN_MLPWeightFile* inwfile = NULL;
	float min, max;
	if (verbose>0)
	{
	    QN_OUTPUT("Loading weights...");
	}
	if (strcmp(config.init_weight_format, "matlab")==0)
	{
	    inwfile = new QN_MLPWeightFile_Matlab(debug, "init_weight_file",
						  init_weight_fp,
						  QN_READ,
						  mlp_layers,
						  mlp_layer_size);
	}
	else if (strcmp(config.init_weight_format, "rap3")==0)
	{
	    // Only support RAP weight files for 3 layer MLPs.
	    if (mlp_layers==3)
	    {
		inwfile = new QN_MLPWeightFile_RAP3(debug, init_weight_fp,
						    QN_READ,
						    "init_weight_file",
						    mlp_layer_size[0],
						    mlp_layer_size[1],
						    mlp_layer_size[2]);
	    }
	    else
	    {
		QN_ERROR(NULL, "init_weight_format can only be 'rap3' "
			 "format for 3 layer MLPs.");
	    }
	}
	else
	    QN_ERROR(NULL, "unknown init_weight_format '%s'.",
		     config.init_weight_format);
	QN_read_weights(*inwfile, *mlp, &min, &max, debug, "init_weight_file", config.mbt_num_basis);
	QN_OUTPUT("Weights loaded from file, min=%g max=%g.",
		  min, max);
	if (inwfile!=NULL)
	    delete inwfile;
    }

    const char* log_weight_file = config.log_weight_file;
    enum QN_WeightFileType log_weight_type;
    if (strcmp(config.log_weight_format, "matlab")==0)
    {
	log_weight_type = QN_WEIGHTFILE_MATLAB;
    }
    else if (strcmp(config.log_weight_format, "rap3")==0)
    {
	if (mlp_layers==3)
	    log_weight_type = QN_WEIGHTFILE_RAP3;
	else
	    QN_ERROR(NULL, "log_weight_format can only be 'rap3' for "
		     "3 layer MLPs");
    }
    else
	QN_ERROR(NULL, "unknown log_weight_format '%s'.",
		 config.log_weight_format);

    const char* ckpt_weight_file = config.ckpt_weight_file;
    enum QN_WeightFileType ckpt_weight_type;
    if (strcmp(config.ckpt_weight_format, "matlab")==0)
    {
	ckpt_weight_type = QN_WEIGHTFILE_MATLAB;
    }
    else if (strcmp(config.ckpt_weight_format, "rap3")==0)
    {
	if (mlp_layers==3)
	    ckpt_weight_type = QN_WEIGHTFILE_RAP3;
	else
	    QN_ERROR(NULL, "ckpt_weight_format can only be 'rap3' for "
		     "3 layer MLPs");
    }
    else
	QN_ERROR(NULL, "unknown ckpt_weight_format '%s'.",
		 config.ckpt_weight_format);
    size_t train_chunk_size;	// The number of presentations read
				// at one time.
    size_t mlp_bunch_size = config.mlp_bunch_size;
    if (mlp_bunch_size>1)
    {
	train_chunk_size = mlp_bunch_size;
    }
    else
	train_chunk_size = 16;	// By default, use a size of 16.
    unsigned long ckpt_seconds = config.ckpt_hours * 3600;
    // Sort out ouptut weight file here so we don't fail after a big
    // training.
    QN_MLPWeightFile* outwfile;
    if (strcmp(config.out_weight_format, "matlab")==0)
    {
	outwfile = new QN_MLPWeightFile_Matlab(debug, "out_weight_file",
					       out_weight_fp,
					       QN_WRITE,
					       mlp_layers,
					       mlp_layer_size, config.mbt_num_basis);
    }
    else if (strcmp(config.out_weight_format, "rap3")==0)
    {
	// Only support RAP weight files for 3 layer MLPs.
	if (mlp_layers==3)
	{
	    outwfile = new QN_MLPWeightFile_RAP3(debug, out_weight_fp,
						 QN_WRITE,
						 "out_weight_file",
						 mlp_layer_size[0],
						 mlp_layer_size[1],
						 mlp_layer_size[2]);
	}
	else
	{
	    QN_ERROR(NULL, "out_weight_format can only be in 'rap3' "
		     "format for 3 layer MLPs.");
	}
    }
    else
	QN_ERROR(NULL, "unknown out_weight_format '%s'.",
		 config.out_weight_format);

    //cz277 - outmap
    struct MapStruct mapclass;
    if (QN_loadmap(config.map_file_path, config.map_opt, &mapclass) < 0) {
	QN_ERROR(NULL, "bad map_file_path '%s'.", config.map_file_path);
    } 

    
    if (hardtarget_train_str!=NULL)
    {
	assert(hardtarget_cv_str!=NULL);
        
        //cw564 - mbt
        map< string, float* > spkr_wgt = gen_spkwgt(config.mbt_init_lambda_file);
        map< int, string > seg2spkr = gen_seg2spk(config.mbt_seg2spkr_file);
        MBT_Params mbt_params(seg2spkr, spkr_wgt, config.mbt_num_basis);
	QN_HardSentTrainer* trainer =
	    new QN_HardSentTrainer(debug,               // Debugging level.
				   "trainer",           // Debugging tag.
				   verbose,	            // Verbosity level.
				   mlp,                 // MLP.
				   ftrfile_train_str,   // Training ftr strm.
				   hardtarget_train_str, // Training label str.
				   ftrfile_cv_str,      // CV feature stream.
				   hardtarget_cv_str,   // CV label stream.
				   lr_schedule,	    // Learning rate scheduler.
				   0.0,		    // Low target.
				   1.0,		    // High target.
				   log_weight_file, // Where we log weights.
				   log_weight_type, // Format.
				   ckpt_weight_file, // Where we checkpoint.
				   ckpt_weight_type,
				   ckpt_seconds,
				   train_chunk_size, // Batch size.
				   lastlab_reject,  // Allow untrainable frames
				   lrmultipliers,         // Per-section LR scales.
                                   &mbt_params   //cw564 - mbt
			       );
	trainer->train(config.lr_ctr, &mapclass);	//cz277 - outmap
	delete trainer;
    }
    else
    {
	assert(softtarget_train_str!=NULL);
	assert(softtarget_cv_str!=NULL);

	QN_SoftSentTrainer* trainer =
	    new QN_SoftSentTrainer(debug,               // Debugging level.
				   "trainer",           // Debugging tag.
				   verbose,	            // Verbosity level.
				   mlp,                 // MLP.
				   ftrfile_train_str,   // Training ftr strm.
				   softtarget_train_str, // Training label str.
				   ftrfile_cv_str,      // CV feature stream.
				   softtarget_cv_str,   // CV label stream.
				   lr_schedule,	    // Learning rate scheduler.
				   0.0,		    // Low target.
				   1.0,		    // High target.
				   log_weight_file, // Where we log weights.
				   log_weight_type, // Format.
				   ckpt_weight_file,  // Where we checkpoint.
				   ckpt_weight_type,
				   ckpt_seconds,
				   train_chunk_size, // Batch size.
				   lrmultipliers
			       );
	trainer->train();
	delete trainer;
    }

    {
	if (verbose>0)
	{
	    QN_OUTPUT("Starting to write weights...");
	}
	float min, max;
	QN_write_weights(*outwfile, *mlp, &min, &max, debug,
			 "out_weight_file");
	if (outwfile!=NULL)
	    delete outwfile;
	QN_OUTPUT("Weights written to '%s'.", out_weight_file);
    }

    //cz277 - outmap
    if (mapclass.opt > 0) {
        delete []mapclass.srclist;
	delete []mapclass.tgtlist;
	delete []mapclass.tgtcnts;
    }

    //cz277 - mul actv
    delete []hiddenlayer_types;
    /*delete []config.init_random_bias_min.vals;
    delete []config.init_random_bias_max.vals;
    delete []config.init_random_weight_min.vals;
    delete []config.init_random_weight_max.vals;
    delete []config.learnrate_vals.vals;
    delete []config.mlp_size.vals;
    delete []config.mlp_lrmultiplier.vals;*/
    for (int i = 0; i < config.mlp_hidden_type.count; ++i) {
	delete []config.mlp_hidden_type.vals[i];
    }
    delete []config.mlp_hidden_type.vals;

    delete mlp;

    if (out_weight_fp!=NULL)
	QN_close(out_weight_fp);
    if (init_weight_fp!=NULL)
	QN_close(init_weight_fp);
    if (ftr2_norm_fp!=NULL)
	QN_close(ftr2_norm_fp);
    if (ftr1_norm_fp!=NULL)
	QN_close(ftr1_norm_fp);
    if (hardtarget_fp!=NULL)
    {
	QN_close(hardtarget_fp);
    }
    if (unary_fp!=NULL)
    {
	QN_close(unary_fp);
    }
    QN_close_ftrfiles();
// A note for the logfile.
    time(&now);
    QN_OUTPUT("Program stop: %.24s", ctime(&now));
}

int
main(int argc, const char* argv[])
{
    char* progname;		// The name of the prog - set by QN_initargs.

    FILE* log_fp;
    char log_buf[160];



    set_defaults();
    QN_initargs(&argtab[0], &argc, &argv, &progname);

    // map norm_mode_str to val
    config.ftr1_norm_mode = QN_string_to_norm_const(config.ftr1_norm_mode_str);
    config.ftr2_norm_mode = QN_string_to_norm_const(config.ftr2_norm_mode_str);

    // Seed the random number generator.
    srand48(config.init_random_seed);

    log_fp = QN_open(config.log_file, "w");
    assert(setvbuf(log_fp, log_buf, _IOLBF, sizeof(log_buf))==0);

    QN_printargs(log_fp, progname, &argtab[0]);
    QN_logger = new QN_Logger_Simple(log_fp, stderr, progname);
    
    // Install our own out-of-memory handler if possible.
#ifdef QN_HAVE_SET_NEW_HANDLER
    set_new_handler(QN_new_handler);
#endif

    // Set the math mode
    QN_math_init(config.use_pp, config.use_fe, config.use_blas,
		 config.use_cuda, 1);

    qnmultitrn();

    exit(EXIT_SUCCESS);
}
