#include <bits/stdc++.h>
#include <iostream>
#include <random>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#define SEED 500
#define PAT_LEN 50
#define THREADS 64


using namespace std;

typedef struct {
	char* pat;
	char* txt;
	bool* result;
	int* lps;
	int pat_len;
	int txt_len;
	int tid;
	int n_threads;
} threads_input;


double get_time();
long input(string filename, char** textp);
char* random_string(int length, char* text, int seed, long text_len);
void computeLPSArray(char* pat, int M, int* lps);
void* KMPSearch_t(void *args);

int main(int argc, const char *argv[]) {
	const char* filename;
	char *txt, *pat;
	long text_len;
	bool *result_threads;
	int lps[PAT_LEN];
	double start_threads, end_threads;

	pthread_t threads[THREADS];
	threads_input input_th[THREADS];

	// get input
	if (argc != 2){
        printf("Usage: ./executable text_file\n");
        return 1;
    }

    // read text from file and store it in txt, allocate memory for result
    filename = argv[1];
    text_len = input(filename, &txt);
    result_threads = (bool*) calloc(text_len * sizeof(bool), sizeof(bool));

    // get pattern
    pat = random_string(PAT_LEN, txt, SEED, text_len);

    // compute lps
    computeLPSArray(pat, PAT_LEN, lps);
	

	// initialize input
	for (int i = 0; i<THREADS; i++){
		input_th[i].pat = pat;
		input_th[i].txt = txt;
		input_th[i].result = result_threads;
		input_th[i].lps = lps;
		input_th[i].pat_len = PAT_LEN;
		input_th[i].txt_len = text_len;
		input_th[i].tid = i;
		input_th[i].n_threads = THREADS;
	}


    // start computation
	start_threads = get_time();

	// start threads
	for (int i = 0; i<THREADS; i++)
		pthread_create(&threads[i], NULL, KMPSearch_t, (void*) &input_th[i]);
	

	for (int i = 0; i<THREADS; i++)
		pthread_join(threads[i], NULL);

	end_threads = get_time();

	printf("Total time CPU %d threads: %.5lf\n", THREADS, end_threads - start_threads);

	// write results to file
	/* ofstream myfile;
	myfile.open ("result.txt");
	for (int i = 0; i<text_len; i++){
		myfile << result[i] + 0;
	}
	myfile.close(); */

	// free memory
	cudaFreeHost(txt);
	free(pat);
	free(result_threads);

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
	cudaMallocHost((void**)textp, len);
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

void* KMPSearch_t(void *args){

	threads_input* input_th = (threads_input*) args;
	char* pat = input_th->pat;
	char* txt = input_th->txt;
	bool* result = input_th->result;
	int* lps = input_th->lps;
	int pat_len = input_th->pat_len;
	int txt_len = input_th->txt_len;
	int tid = input_th->tid;
	int n_threads = input_th->n_threads;

	int i; // index for txt[]
	int j = 0; // index for pat[]

	int start, end;

	start = tid * (txt_len / n_threads + 1);
	end = (tid + 1) * (txt_len / n_threads + 1) + pat_len - 1;

	if(end > txt_len)
		end = txt_len;

	i = start;

	while (i < end) {
		
		if (pat[j] == txt[i]) {
			j++;
			i++;
		}

		if (j == pat_len) {
			result[i - j] = true;
			j = lps[j - 1];
		}

		// mismatch after j matches
		else if (i < txt_len && pat[j] != txt[i]) {
			// Do not match lps[0..lps[j-1]] characters,
			// they will match anyway
			if (j != 0)
				j = lps[j - 1];
			else
				i = i + 1;
		}
	}

	return NULL;
}