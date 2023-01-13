/*
Name - Animesh Dwivedi (MIT2021078)
compilation flags : pthread
compiler used : gcc
Observations  : For small number of rows, the program runs better for 1-thread than for large threads, due to overhead
				of thread creation and linking. As soon as the number of rows increases, the performance with
				more number of thread increase i.e. lesser execution time.
*/

#include<bits/stdc++.h>
#include<pthread.h>
#include <stdint.h>
#include <sys/time.h>

using namespace std;

struct thread_rows{
    int start;
    int end;
};
int nrows,ncoloumns,nvalues;
vector<int> rows;
vector<int> coloumns;
vector<int> values;
vector<int> ans;
vector<int> input_vector;
void *sparse_vector_multiplication(void *arg) {
    struct range * ranges = (struct range *) arg;
    for(int i=ranges->start; i < ranges->end; i++) {
               ans[rows[i]]+=values[i]*input_vector[coloumns[i]];
    }
    return NULL;
}

int main(){
	std::ifstream inputFile1;
    inputFile1.open("matrix.csr.mtx",ios::in);
    
    inputFile1 >> nrows >> ncoloumns >> nvalues;
    rows=vector<int>(nrows,0);
    coloumn=vector<int>(ncoloumns,0);
    values=vector<int> (nvalues,0);
    
 	for(int i=0;i<nvalues;i++){
 		InputFile1 >> rows[i] >> coloumns[i] >> values[i];
 	}
 	
    ifstream inputFile2;
    inputFile2.open("vector.txt", ios::in);
    
    input_vector=vector<int>(ncoloumns,0);
    for(int i=0;i<ncoloumns;i++){
    	InputFile2 >> input_vector[i];
    }
    
    // sparse matrix * vector multiplication in sequential
    
    clock_t start, end, time_taken_sequential;
    start=clock();
    for(int i=0;i<nvalues;i++){
    	ans[rows[i]]+=values[i]*input_vector[coloumns[i]];
    }
    end=clock();
    time_taken_sequential=end-start;
    ans.clear();
    double time_taken_by_sequential = ((double)time_taken_sequential)/CLOCKS_PER_SEC;
    cout<<"Time taken by the sequential "<< time_taken_by_sequential <<" seconds.\n";
     
    // sparse matrix * vector multipplication in parallel  
    for(int threads=1;threads<=64;threads*=2){
    	pthread_t thread_list[threads];
		struct thread_rows th_rows[threads];

		int start_, range;
		start_ = 0;
	
		range = nrows / threads;    
		for(int i = 0; i < threads; i++) 
		{
			th_rows[i].start = start_;
			th_rows[i].end = start_ + range;
			start_ += range;
		}
		th_rows[threads-1].end = nrows;
		
		clock_t start, end, time_taken_parallel;
	    start = clock();

		for(int i = 0; i < threads; i++) {
			pthread_create(&thread_list[i], NULL, sparse_vector_multiplication, &th_rows[i]);
		}	
		for(int i = 0; i < threads; i++) {
			pthread_join(thread_list[i], NULL);
		}
		
		end = clock();
		time_taken_parallel = end - start;
		
		double time_taken_by_parallel = ((double)time_taken_parallel)/CLOCKS_PER_SEC; // in seconds
	    cout<<"in parallel with :"<< threads<<" threads time taken : " << time_taken_by_parallel << "\n";
	    ans.clear();
    }
}

