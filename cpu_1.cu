#include <bits/stdc++.h>
#include <iostream>
#include <random>
#include <time.h>
#include <sys/time.h>

#define SEED 500
#define PAT_LEN 50

using namespace std;


double get_time();
long input(string filename, char** textp);
char* random_string(int length, char* text, int seed, long text_len);
void computeLPSArray(char* pat, int M, int* lps);
void KMPSearch(char* pat, char* txt, bool* result, int* lps, int pat_len, int txt_len);
void* KMPSearch_t(void *args);

int main(int argc, const char *argv[]) {
	const char* filename;
	char *txt, *pat;
	long text_len;
	bool *result;
	int lps[PAT_LEN];
	double start_cpu, end_cpu;

	// get input
	if (argc != 2){
        printf("Usage: ./executable text_file\n");
        return 1;
    }

    // read text from file and store it in txt, allocate memory for result
    filename = argv[1];
    text_len = input(filename, &txt);
    result = (bool*) calloc(text_len * sizeof(bool), sizeof(bool));

    // get pattern
    pat = random_string(PAT_LEN, txt, SEED, text_len);


    // compute lps
    computeLPSArray(pat, PAT_LEN, lps);

    // start computation
	start_cpu = get_time();
	KMPSearch(pat, txt, result, lps, PAT_LEN, text_len);
	end_cpu = get_time();
	printf("Total time CPU single thread: %.5lf\n", end_cpu - start_cpu);

	// free memory
	cudaFreeHost(txt);
	free(pat);
	free(result);

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

// code adapted from https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/
void KMPSearch(char* pat, char* txt, bool* result, int* lps, int pat_len, int txt_len){
	int i = 0; // index for txt[]
	int j = 0; // index for pat[]

	while ((txt_len - i) >= (pat_len - j)) {
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
}
