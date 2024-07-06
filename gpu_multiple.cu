#include <bits/stdc++.h>
#include <iostream>
#include <random>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <thread>

#define SEED 500
#define DEV_NUM 0
#define STREAM_MAX_NUM 4

#define PAT_LEN 50
#define PAT_AMNT 4

#define BITES_SCANNED_PER_STREAM 1024 * 4096
#define THREADN 128
#define BLOCKN 8

#define CONCURRENT_BUFFERS 4

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

using namespace std;

typedef struct {
	long* result_indexes;
	int result_amount;
} stream_results;

typedef struct {
	char* txt_d;
	char* pat_d;
	int* lps_d;
	stream_results* results;
	int* result_indexes_d;
	int* result_amounts_d;
	int* result_indexes;
	int* result_amounts;
	cudaStream_t stream;
	long txt_len;
	int pat_len;
	int result_len;
	int shared_memory;
	long start;
	mutex* launching;
} threads_input;

double get_time();
long input(string filename, char** textp);
char* random_string(int length, char* text, long text_len);
void computeLPSArray(char* pat, int M, int* lps);
void * manage_kernel_call(void *input);
__global__ void KMPSearch_d(const char* pat, const char* txt, const int* lps, int* result_indices, int* result_amounts, const int max_res_per_thread, int pat_len, int txt_len, int n_threads, int n_blocks);
void adjust_results(int* result_indexes, int* result_amounts, stream_results* results, int max_res_per_thread, long offset, int n_threads, int n_blocks);

int main(int argc, const char *argv[]) {
	const char* filename;
	char *txt, *pats[PAT_AMNT];
	long text_len;
	int* lpses[PAT_AMNT];
	double start_gpu, end_gpu;
    size_t free_mem, total_mem;
    int streams_num;

	cudaStream_t cpy_h2d_stream;

	char *txts_d[CONCURRENT_BUFFERS];
	char *pats_d[PAT_AMNT];
	int *lpses_d[PAT_AMNT];
	int shared_memory = PAT_LEN*sizeof(char) + PAT_LEN*sizeof(int);
	vector<pthread_t> threads[CONCURRENT_BUFFERS];

	int i;
	

    // get input
    if (argc != 2){
        printf("Usage: ./executable text_file");
        return 0;
    }

    filename = argv[1];
    text_len = input(filename, &txt);

    // get patterns
	srand(SEED);
	for (i = 0; i<PAT_AMNT; i++)
	    pats[i] = random_string(PAT_LEN, txt, text_len);


    // compute lpses
	for (i = 0; i<PAT_AMNT; i++){
		lpses[i] = (int*) malloc(PAT_LEN * sizeof(int));
		computeLPSArray(pats[i], PAT_LEN, lpses[i]);
	}

    //############ calculate parameters ############ 
	// number of streams needed, we need at least one stream unused by patterns to have the host to device copy on a separate untouched stream
	streams_num = min(PAT_AMNT, STREAM_MAX_NUM - 1);

	// need to see if it all fits in memory
	// the space is occupied by txts, results, patterns, lpses
	// concurrent_buffers * BITES_SCANNED_PER_STREAM + PAT_LEN * 2 * PAT_AMNT + result_indexes + result_amounts
	// first check that concurrent_buffers * BITES_SCANNED_PER_STREAM  fits in memory, otherwise throw error
	// then check how many patterns fit in memory

    // get device available memory
    cudaMemGetInfo(&free_mem, &total_mem); 
    free_mem = free_mem * 0.9; // use only 90% of the available memory

    int txts_bytes_amnt = CONCURRENT_BUFFERS * (BITES_SCANNED_PER_STREAM + PAT_LEN - 1);
	int pats_bytes_amnt = shared_memory;

	int max_result_space = 0;
	for (i = 0; i<PAT_AMNT; i++)
		max_result_space = max(max_result_space, BITES_SCANNED_PER_STREAM / (PAT_LEN - lpses[i][PAT_LEN - 1]) / (BLOCKN * THREADN) + 1);
	
	int results_bytes_amnt = max_result_space * BLOCKN * THREADN * streams_num * sizeof(int);
	int indices_bytes_amnt = BLOCKN * THREADN * streams_num * sizeof(int);

	if (txts_bytes_amnt + max_result_space + indices_bytes_amnt > free_mem){
		printf("Error: Not enough memory for the texts buffers and results on device, reduce BITES_SCANNED_PER_STREAM, CONCURRENT_BUFFERS or STREAM_MAX_NUM\n");
		return 1;
	}

	int max_patterns = (free_mem - txts_bytes_amnt - max_result_space - indices_bytes_amnt) / pats_bytes_amnt;

	if (max_patterns < PAT_AMNT){
		printf("Error: Not enough memory for the patterns on device, reduce PAT_AMNT\n");
		return 1;
	}

	// number of txts chunks needed to scan the text
	// need to cast the division to avoid integer division overflow
    float a =  ((float) text_len) / ((float) BITES_SCANNED_PER_STREAM );
    int txts_chunks_amnt = ceil(a);

	//############ Allocate space ############
	cudaStream_t streams[streams_num];

	int *result_indexes_d_array[streams_num];
	int *result_amounts_d_array[streams_num];
	int *result_indexes_array[streams_num];
	int *result_amounts_array[streams_num];
	mutex launching[streams_num];

	stream_results results[PAT_AMNT][txts_chunks_amnt];

	// allocate space for the patterns and lpses, copy them to device
	for (i = 0; i < PAT_AMNT; i++){
		CHECK(cudaMalloc((void**)&pats_d[i], PAT_LEN*sizeof(char)));
		CHECK(cudaMalloc((void**)&lpses_d[i], PAT_LEN*sizeof(int)));
		CHECK(cudaMemcpy(pats_d[i], pats[i], PAT_LEN*sizeof(char), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(lpses_d[i], lpses[i], PAT_LEN*sizeof(int), cudaMemcpyHostToDevice));
	}

	// allocate space for the txts
	for (i = 0; i<CONCURRENT_BUFFERS; i++)
		CHECK(cudaMalloc((void**)&txts_d[i], txts_bytes_amnt / CONCURRENT_BUFFERS));

	// TODO: attach indexes and amount for a faster copy back
	// allocate space for the results, intialize launching mutexes
	cudaStreamCreate(&cpy_h2d_stream);
	for (i = 0; i<streams_num; i++){
		cudaStreamCreate(&streams[i]);
        
		CHECK(cudaMalloc((void**)&result_indexes_d_array[i], results_bytes_amnt / streams_num));
		CHECK(cudaMalloc((void**)&result_amounts_d_array[i], indices_bytes_amnt/ streams_num));

		CHECK(cudaMallocHost((void**)&result_indexes_array[i], results_bytes_amnt / streams_num));
		CHECK(cudaMallocHost((void**)&result_amounts_array[i], indices_bytes_amnt/ streams_num));
	}
	
	//############ initial copies ############
	// copy the patterns and lpses
	for (i = 0; i < PAT_AMNT; i++){
		CHECK(cudaMemcpyAsync(pats_d[i], pats[i], PAT_LEN*sizeof(char), cudaMemcpyHostToDevice, cpy_h2d_stream));
		CHECK(cudaMemcpyAsync(lpses_d[i], lpses[i], PAT_LEN*sizeof(int), cudaMemcpyHostToDevice, cpy_h2d_stream));
	}

	//############ start computation ############
	CHECK(cudaDeviceSynchronize());
	start_gpu = get_time();

	for (long txt_ind = 0; txt_ind < txts_chunks_amnt; txt_ind++){

		int buf_ind = txt_ind % CONCURRENT_BUFFERS; 
		// for every thread in threads[buf_ind] wait for it to finish
		
		for (pthread_t thread : threads[buf_ind]){
			pthread_join(thread, NULL);
		}
		threads[buf_ind].clear();


		long start = txt_ind * txts_bytes_amnt / CONCURRENT_BUFFERS;
		long len = min((txt_ind + 1) * txts_bytes_amnt / CONCURRENT_BUFFERS, text_len) - start;
		// copy the text to device
		CHECK(cudaMemcpyAsync(txts_d[buf_ind], txt + start, len, cudaMemcpyHostToDevice, cpy_h2d_stream));
		CHECK(cudaStreamSynchronize(cpy_h2d_stream));

		// launch the threads for each pattern
		for (int pat_ind = 0; pat_ind < PAT_AMNT; pat_ind++){

			int stream_ind = (txt_ind*PAT_AMNT + pat_ind) % streams_num;
 
			threads_input* input_th = (threads_input*) malloc(sizeof(threads_input));
			input_th->txt_d = txts_d[buf_ind];
			input_th->pat_d = pats_d[pat_ind];
			input_th->lps_d = lpses_d[pat_ind];
			input_th->results = &results[pat_ind][txt_ind];
			input_th->result_indexes_d = result_indexes_d_array[stream_ind];
			input_th->result_amounts_d = result_amounts_d_array[stream_ind];
			input_th->result_indexes = result_indexes_array[stream_ind];
			input_th->result_amounts = result_amounts_array[stream_ind];
			input_th->stream = streams[stream_ind];
			input_th->txt_len = len;
			input_th->pat_len = PAT_LEN;
			input_th->result_len = results_bytes_amnt / streams_num / sizeof(int);
			input_th->shared_memory = shared_memory;
			input_th->start = start;
			input_th->launching = &launching[stream_ind];

			pthread_t thread;

			// launch the thread that collects the results
			pthread_create(&thread, NULL, manage_kernel_call, input_th);
			threads[buf_ind].push_back(thread);
		}

	}

	for (i = 0; i<CONCURRENT_BUFFERS; i++){
		for (pthread_t thread : threads[i]){
			pthread_join(thread, NULL);
		}
	}

	CHECK(cudaDeviceSynchronize());
	end_gpu = get_time();

	printf("Total time GPU for %d patterns: %.5lf\n", PAT_AMNT, end_gpu - start_gpu);

	
	//############ end computation ############

	cudaFreeHost(txt);
	cudaStreamDestroy(cpy_h2d_stream);
	for (i = 0; i<streams_num; i++){
		cudaStreamDestroy(streams[i]);
		
		CHECK(cudaFree(result_indexes_d_array[i]));
		CHECK(cudaFree(result_amounts_d_array[i]));

		CHECK(cudaFreeHost(result_indexes_array[i]));
		CHECK(cudaFreeHost(result_amounts_array[i]));
	}

    for (i = 0; i<PAT_AMNT; i++){
        cudaFree(pats_d[i]);
        cudaFree(lpses_d[i]);
        free(pats[i]);
        free(lpses[i]);
    }

    for (i = 0; i<CONCURRENT_BUFFERS; i++){
        cudaFree(txts_d[i]);
    }

    return 0; 
}

double get_time() { // function to get the time of day in second
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

char* random_string(int length, char* text, long text_len){
	char* result = (char*) malloc(length);
	long start_index = rand() % (text_len - length);
	for (long i = 0; i< length; i++)
		result[i] = text[start_index + i];

	return result;
}

long input(string filename, char** textp){
	ifstream file(filename);
	string str;
	string file_contents;
	getline(file, str);
	
	while (getline(file, str)){
		str.pop_back();
		for (int i = 0; i< str.length(); i++){
			str[i] = toupper(str[i]);
		}
		file_contents += str;
	}
	long len = file_contents.length();
	CHECK(cudaMallocHost((void**)textp, len));
	strcpy (*textp, file_contents.c_str());
	return file_contents.length();
}

// code taken from https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/
void computeLPSArray(char* pat, int M, int* lps){
	// length of the previous longest prefix suffix
	int len = 0;

	lps[0] = 0; // lps[0] is always 0

	// the loop calculates lps[i] for i = 1 to M-1
	int i = 1;
	while (i < M) {
		if (pat[i] == pat[len]) {
			len++;
			lps[i] = len;
			i++;
		}
		else // (pat[i] != pat[len])
		{
			// This is tricky. Consider the example.
			// AAACAAAA and i = 7. The idea is similar
			// to search step.
			if (len != 0) {
				len = lps[len - 1];

				// Also, note that we do not increment
				// i here
			}
			else // if (len == 0)
			{
				lps[i] = 0;
				i++;
			}
		}
	}
}


void * manage_kernel_call(void *input){
	threads_input *in = (threads_input*) input;
	char* txt_d = in->txt_d;
	char* pat_d = in->pat_d;
	int* lps_d = in->lps_d;
	stream_results* results= in->results;
	int* result_indexes_d = in->result_indexes_d;
	int* result_amounts_d = in->result_amounts_d;
	int* result_indexes = in->result_indexes;
	int* result_amounts = in->result_amounts;
	cudaStream_t stream = in->stream;
	long txt_len = in->txt_len;
	int pat_len = in->pat_len;
	int result_len = in->result_len;
	int shared_memory = in->shared_memory;
	mutex *launching = in->launching;
	
	long start = in->start;


	int max_res_per_thread = result_len / (BLOCKN * THREADN);


	// launch the kernel
	(*launching).lock();
	KMPSearch_d<<<BLOCKN, THREADN, shared_memory, stream>>>(pat_d, txt_d, lps_d, result_indexes_d, result_amounts_d, max_res_per_thread, pat_len, txt_len, THREADN, BLOCKN);
	CHECK_KERNELCALL();	

	// copy the results back
	CHECK(cudaMemcpyAsync(result_indexes, result_indexes_d, result_len * sizeof(int), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(result_amounts, result_amounts_d, (BLOCKN * THREADN) * sizeof(int), cudaMemcpyDeviceToHost, stream));

	// wait for the results to be copied back
	CHECK(cudaStreamSynchronize(stream));


	// adjust the results
	adjust_results(result_indexes, result_amounts, results, max_res_per_thread, start, THREADN, BLOCKN);

	// mark the stream as finished
	(*launching).unlock();

	free(input);
	return NULL;
}

__global__ void KMPSearch_d(const char* pat, const char* txt, const int* lps, int* result_indices, int* result_amounts, const int max_res_per_thread, int pat_len, int txt_len, int n_threads, int n_blocks){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ int lps_sh[];
	char *pat_sh = (char*) &lps_sh[pat_len];
	uint start, end, j = 0, i;
	bool match, pat_ind_zero;
	int result_amount = 0;
	int result_offset = tid*max_res_per_thread;

	// copy the lps and pat to shared memory
	for (i = 0; i< pat_len; i+= n_threads){
		lps_sh[(tid + i)% pat_len] = lps[(tid + i)% pat_len];
		pat_sh[(tid + i)% pat_len] = pat[(tid + i)% pat_len];
	}


	// calculate start and end so that txt is covered and ther is pat_len-1 overlap
	start = tid * (txt_len / (n_threads * n_blocks) + 1);
	end = (tid + 1) * (txt_len / (n_threads * n_blocks) + 1) + pat_len - 1;

	if(end > txt_len)
		end = txt_len;

	i = start;

	while(i < end){
		match = false, pat_ind_zero = false;

		/////////////////////// if match ///////////////////////
		if( pat_sh[j] == txt[i]){
			i ++;
			j ++;
		}

		/////////////////////// if found branch ///////////////////////
		pat_ind_zero = (j == 0);

		if(j == pat_len){
			result_indices[result_offset + result_amount] = i - j;
			result_amount ++;
			j = lps_sh[j - 1];
		}

		/////////////////////// if mismatch branch ///////////////////////
		// in original code this is an else if. It should not be a problem to not have an 
		// else, since i and j 

		pat_ind_zero = (j == 0);

		match = 0;
		if(i < end)
			match = (pat_sh[j] == txt[i]);
		j = j*match + !match*!pat_ind_zero*lps_sh[j - 1 +pat_ind_zero];
		//if not found and j!=0
		
		i += !match*pat_ind_zero; // if there is a mismatch and j==0: i++
	}
	result_amounts[tid] = result_amount;
}

void adjust_results(int* result_indexes, int* result_amounts, stream_results* results, int max_res_per_thread, long offset, int n_threads, int n_blocks){
	int total_results = 0;
	int i, j, k = 0;

    // aggregate the number of results found by each thread
	for (i = 0; i< n_threads * n_blocks; i++){
		total_results += result_amounts[i];
		if(result_amounts[i] > max_res_per_thread){
			printf("Error: result_amounts[%d] = %d > max_res_per_thread %d\n", i, result_amounts[i], max_res_per_thread);

		}
		if(result_amounts[i] < 0){
			printf("Error: result_amounts[%d] = %d < 0\n", i, result_amounts[i]);
		}
	}
	
	results->result_amount = total_results;
	


	if (total_results == 0){
		results->result_indexes = NULL;
		return;
	}

    // allocate memory for the results
	results->result_indexes = (long*) malloc(total_results * sizeof(long));

    
    // store the actual indexes of the results
	for (i = 0; i< n_threads * n_blocks; i++){
		for (j = 0; j< result_amounts[i]; j++){

			results->result_indexes[k] = offset + ((long)result_indexes[i * max_res_per_thread + j]);
			k++;
		}
	}
	if(k != total_results){
		printf("Error: k != total_results\n");
	}
}