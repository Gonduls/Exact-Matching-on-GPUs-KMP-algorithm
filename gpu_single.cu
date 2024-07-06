#include <bits/stdc++.h>
#include <iostream>
#include <random>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#define SEED 500

#define STREAM_MAX_NUM 32
#define BITES_SCANNED_PER_STREAM 1024*256
#define THREADN 32
#define BLOCKN 128
#define PAT_LEN 50
#define DEV_NUM 0

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
	char* txt; 
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
	int n_streams;
	int tid;
} threads_input;

double get_time();
long input(string filename, char** textp);
char* random_string(int length, char* text, int seed, long text_len);
void computeLPSArray(char* pat, int M, int* lps);
void * manage_stream(void *input);
__global__ void KMPSearch_d(const char* pat, const char* txt, const int* lps, int* result_indices, int* result_amounts, const int max_res_per_thread, int pat_len, int txt_len, int n_threads, int n_blocks);
void adjust_results(int* result_indexes, int* result_amounts, stream_results* results, int max_res_per_thread, long offset, int chunk_id, int n_threads, int n_blocks);

int main(int argc, const char *argv[]) {
	const char* filename;
	char *txt, *pat;
	long text_len;
	int lps[PAT_LEN];
	double start_gpu, end_gpu;
    size_t free_mem, total_mem;
    int streams_num, max_streams_num, streams_needed;
    long bytes_occupied_by_stream;

	cudaStream_t *streams;

	char *pat_d, **txts_d;
	int *lps_d;
	int **result_indexes_d_array, **result_amounts_d_array;
	int **result_indexes_array, **result_amounts_array;
	int shared_memory = PAT_LEN*sizeof(char) + PAT_LEN*sizeof(int);

    // get input
    if (argc != 2){
        printf("Usage: ./executable text_file");
        return 0;
    }

    filename = argv[1];
    text_len = input(filename, &txt);

    // get pattern
    pat = random_string(PAT_LEN, txt, SEED, text_len);

    // print the pattern
    /* for (int i = 0; i<PAT_LEN; i++)
        printf("%c", pat[i]);
    printf("\n"); */

    // compute lps
    computeLPSArray(pat, PAT_LEN, lps);
    
    //############ calculate parameters ############

    // get device available memory
    cudaMemGetInfo(&free_mem, &total_mem); 
    printf("Total memory: %ld\n", total_mem);
    free_mem = free_mem * 0.9; // use only 90% of the available memory
    printf("Using memory: %ld\n", free_mem);

    // number of streams needed to scan the text
	// need to cast the division to avoid integer division overflow
    float a =  ((float) text_len) / ((float) BITES_SCANNED_PER_STREAM );
    streams_needed = ceil(a);
    
    // calculate the number of streams that can be run concurrently
	// for each stream we need to allocate memory for the text and the results
	// The text scanned is composed of BITES_SCANNED_PER_STREAM + an overlap of PAT_LEN - 1
    bytes_occupied_by_stream = (BITES_SCANNED_PER_STREAM + PAT_LEN - 1) *2 ;
    max_streams_num = free_mem / bytes_occupied_by_stream;

	streams_num = min(streams_needed, max_streams_num);
	streams_num = min(streams_num, STREAM_MAX_NUM);

	printf("Streams to be run: %d\n", streams_num);
	printf("Streams needed: %d\n", streams_needed);

	// Calculate the results length for each stream
	// The length should be a multiple of BLOCKN * THREADN
	int result_per_thread = BITES_SCANNED_PER_STREAM / (PAT_LEN - lps[PAT_LEN - 1]) / (BLOCKN * THREADN) + 1;
	int result_len = result_per_thread * BLOCKN * THREADN;

	//############ Allocate space ############
	txts_d = (char**) malloc(streams_num * sizeof(char*));
	streams = (cudaStream_t*) malloc(streams_num * sizeof(cudaStream_t));
	result_indexes_d_array = (int**) malloc(streams_num * sizeof(int*));
	result_amounts_d_array = (int**) malloc(streams_num * sizeof(int*));
	result_indexes_array = (int**) malloc(streams_num * sizeof(int*));
	result_amounts_array = (int**) malloc(streams_num * sizeof(int*));
	
	CHECK(cudaMalloc((void**)&pat_d, PAT_LEN*sizeof(char)));
	CHECK(cudaMalloc((void**)&lps_d, PAT_LEN*sizeof(int)));
	CHECK(cudaMemcpy(pat_d, pat, PAT_LEN*sizeof(char), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(lps_d, lps, PAT_LEN*sizeof(int), cudaMemcpyHostToDevice));
	


	// one time allocations and copies
	for (int i = 0; i<streams_num; i++){
		cudaStreamCreate(&streams[i]);
        CHECK(cudaMalloc((void**)&txts_d[i], (BITES_SCANNED_PER_STREAM + PAT_LEN - 1) * sizeof(char)));
		CHECK(cudaMalloc((void**)&result_indexes_d_array[i], result_len * sizeof(int)));
		CHECK(cudaMalloc((void**)&result_amounts_d_array[i], BLOCKN * THREADN * sizeof(int)));

		CHECK(cudaMallocHost((void**)&result_indexes_array[i], result_len * sizeof(int)));
		CHECK(cudaMallocHost((void**)&result_amounts_array[i], BLOCKN * THREADN * sizeof(int)));
	}

	pthread_t threads[streams_num];
	threads_input input_th[streams_num];
	
	stream_results results[streams_needed];

	for (int i = 0; i<streams_num; i++){
		input_th[i].txt = txt;
		input_th[i].txt_d = txts_d[i];
		input_th[i].pat_d = pat_d;
		input_th[i].lps_d = lps_d;
		input_th[i].results = results;
		input_th[i].result_indexes_d = result_indexes_d_array[i];
		input_th[i].result_amounts_d = result_amounts_d_array[i];
		input_th[i].result_indexes = result_indexes_array[i];
		input_th[i].result_amounts = result_amounts_array[i];
		input_th[i].stream = streams[i];
		input_th[i].txt_len = text_len;
		input_th[i].pat_len = PAT_LEN;
		input_th[i].result_len = result_len;
		input_th[i].shared_memory = shared_memory;
		input_th[i].n_streams = streams_num;
		input_th[i].tid = i;
    }

	//############ start computation ############

	CHECK(cudaDeviceSynchronize());
	start_gpu = get_time();
	for (int i = 0; i<streams_num; i++){
		pthread_create(&threads[i], NULL, manage_stream, &input_th[i]);
	}

	for (int i = 0; i<streams_num; i++){
		pthread_join(threads[i], NULL);
	}

	CHECK(cudaDeviceSynchronize());
	end_gpu = get_time();

	printf("Total time GPU: %.5lf\n", end_gpu - start_gpu);


	//############ write results to a file ############
    // uncomment the following code to write the results to a file

    /* 
	// interpret the results and create a result file with 0 and 1
	// write the boolean array results to file named as the input file + "_result" as a string of 0 and 1
	string output_filename = filename;
	output_filename += "_result_gpu_indexes.txt";
	ofstream output_file(output_filename);

	long index = 0;
	for (int i = 0; i<streams_needed; i++){
		//printf("results[i].result_amount %d\n", results[i].result_amount);
		for (int j = 0; j<results[i].result_amount; j++){
			//printf("Result index: %ld\n", results[i].result_indexes[j]);
			while(index < results[i].result_indexes[j]){
				output_file << 0;
				index ++;
			}
			output_file << 1;
			index ++;
		}
	}
	while(index < text_len){
		output_file << 0;
		index ++;
	} 
    */

    
	// free memory
	for (int i = 0; i<streams_num; i++){
		cudaStreamDestroy(streams[i]);
		cudaFree(txts_d[i]);
		cudaFree(result_indexes_d_array[i]);
		cudaFree(result_amounts_d_array[i]);
		cudaFreeHost(result_indexes_array[i]);
		cudaFreeHost(result_amounts_array[i]);
	}
	cudaFreeHost(txt);

    for (int i = 0; i<streams_needed; i++)
        free(results[i].result_indexes);
    
	return 0; 
}

double get_time() { // function to get the time of day in second
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

char* random_string(int length, char* text, int seed, long text_len){
	char* result = (char*) malloc(length);
	srand(seed);
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

void * manage_stream(void *input){
	threads_input *in = (threads_input*) input;
	char* txt = in->txt;
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
	int n_streams = in->n_streams;
	int tid = in->tid;

	long i;
	int max_res_per_thread = result_len / (BLOCKN * THREADN);

	for(i=tid; (i - 1)*BITES_SCANNED_PER_STREAM < txt_len; i+=n_streams){

		// calculate start and end so that txt is covered and ther is pat_len-1 overlap
		long start = i * BITES_SCANNED_PER_STREAM;
		long end = min((i+1) * BITES_SCANNED_PER_STREAM + pat_len - 1, txt_len);

		if(end < start)
			break;

		// copy the text to device
		CHECK(cudaMemcpyAsync(txt_d, txt + start, (end - start) * sizeof(char), cudaMemcpyHostToDevice, stream));

		KMPSearch_d<<<BLOCKN, THREADN, shared_memory, stream>>>(pat_d, txt_d, lps_d, result_indexes_d, result_amounts_d, max_res_per_thread, pat_len, end - start, THREADN, BLOCKN);
		CHECK_KERNELCALL();


		// copy the results back to host
		CHECK(cudaMemcpyAsync(result_indexes, result_indexes_d, result_len * sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECK(cudaMemcpyAsync(result_amounts, result_amounts_d, BLOCKN * THREADN * sizeof(int), cudaMemcpyDeviceToHost, stream));

		CHECK(cudaStreamSynchronize(stream));

		// adjust the results, aggregation operation
		adjust_results(result_indexes, result_amounts, results, max_res_per_thread, start, i, THREADN, BLOCKN);
	}

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

	// copy lps to shared memory
	// copy pat to shared memory
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

void adjust_results(int* result_indexes, int* result_amounts, stream_results* results, int max_res_per_thread, long offset, int chunk_id, int n_threads, int n_blocks){
	int total_results = 0;
	int i, j, k = 0;

    // aggregate the number of results found by each thread
	for (i = 0; i< n_threads * n_blocks; i++){
		total_results += result_amounts[i];
		if(result_amounts[i] > max_res_per_thread){
			printf("Error: result_amounts[%d] = %d, chunk_id = %d\n", i, result_amounts[i], chunk_id);

		}
		if(result_amounts[i] < 0){
			printf("Error: result_amounts[%d] = %d\n", i, result_amounts[i]);
		}
	}
	
	results[chunk_id].result_amount = total_results;
	


	if (total_results == 0){
		results[chunk_id].result_indexes = NULL;
		return;
	}

    // allocate memory for the results
	results[chunk_id].result_indexes = (long*) malloc(total_results * sizeof(long));


    // store the actual indexes of the results
	for (i = 0; i< n_threads * n_blocks; i++){
		for (j = 0; j< result_amounts[i]; j++){

			results[chunk_id].result_indexes[k] = offset + ((long)result_indexes[i * max_res_per_thread + j]);
			k++;
		}
	}
	if(k != total_results){
		printf("Error: k != total_results\n");
	}
}