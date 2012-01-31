#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <cuda.h>
#include <time.h>

using namespace std;

/*  Cannot use built-in functions, need to rewrite pow function so it can  run on the device, kinda reinventing the wheel over here
	Im not sure if CUDA has something in its SDK, however better build it myself to avoid any overhead
	0^0 even though is not defined, im treating it as 1 
*/
__device__ void calculate_exponent(int base,int exponent,long &result){
	result = 1L;
	if(exponent==0){
		return;
	}
	for(int counter=1;counter<=exponent;counter++){
		result *= (long)base;
	}
}


__global__ void evaluate(int *input, int totalSizeOfArray, int number_genes, int individualsPerThread, int number_blocks, int threads_per_block, long *scores){

	/*global position in population array index calculation */
	int startingPosition = (blockIdx.y * threads_per_block * number_genes * individualsPerThread) + (threadIdx.y * number_genes * individualsPerThread);
	if(startingPosition>=totalSizeOfArray){
		return; //*return if thread is useless, the final block may have some threads that will not compute any data therefore we return early */
	}

	/*global position in scores array index calculation  */
	int startingPosition_scores = (blockIdx.y * threads_per_block * individualsPerThread) + (threadIdx.y * individualsPerThread);


	long acumulated = 0L;
	long temp = 0L;
	for(int counter_individuals=0;counter_individuals<individualsPerThread;counter_individuals++){
		if(startingPosition + (counter_individuals*number_genes) >= totalSizeOfArray){
			return;
		}
		for(int counter_gene=0;counter_gene<number_genes;counter_gene++){
			int base = startingPosition + (counter_individuals*number_genes) + counter_gene;
			calculate_exponent(input[base],(number_genes-1)-counter_gene,temp);
			acumulated += temp;
		}
		scores[startingPosition_scores+counter_individuals] = acumulated;
		
		acumulated=0L;
	}

}

__device__ void determine_fitness_solution(unsigned long desired_number, unsigned long actual_result, unsigned long &deviation){
	if(desired_number>actual_result){
		deviation = desired_number - actual_result;
	}
	if(actual_result>desired_number){
		deviation = actual_result - desired_number;
	}
	if(actual_result==desired_number){
		deviation = 0;
	}
}

__global__ void scan_for_solution(long *scores_array, int number_individuals, int individuals_per_thread, int threads_per_block, int *solution_found_flag, unsigned long desired_number, int acceptable_error){
	int starting_position_in_scores = (blockIdx.y * threads_per_block * individuals_per_thread) + (threadIdx.y * individuals_per_thread);
	if(starting_position_in_scores>=number_individuals){
		return; /* Return if useless thread */
	}
	unsigned long result;
	unsigned long deviation;
	for(int counter_individuals=0;counter_individuals<individuals_per_thread;counter_individuals++){
		if(starting_position_in_scores+counter_individuals>=number_individuals){
			return;
		}
		result = scores_array[starting_position_in_scores+counter_individuals];
		determine_fitness_solution(desired_number,result,deviation);
		if(deviation==0 || deviation<acceptable_error){
			*solution_found_flag = starting_position_in_scores + counter_individuals;
		}
	}
}


	int main(){

		

		/* define settings */
		const unsigned int number_genes = 10;
		const unsigned int number_individuals = 10000000;

		const unsigned int threads_per_block_evaluation = 500; //DO NOT FORGET: BLOCK IS 1 thread width, and threads_per_block  height, MAX 512 
		const unsigned int individuals_per_thread_evaluation = 50;

		/* desired algorithm result and acceptable error */

		const unsigned long desired_number = 123456;
		const unsigned int acceptable_error_window = 1000; /*  So result can be +- acceptable_error_window */

		/* allocate and randomly initialize memory for population */
		int *population_array_host = new int[number_genes*number_individuals];
		int *population_array_device;
		srand ( time(NULL) );
		for(int contador=0;contador<number_genes*number_individuals;contador++){
			population_array_host[contador] = ( rand()  % 10 );
		}
		size_t memory_for_population = number_genes*number_individuals*sizeof(int);
		cudaMalloc((void **) &population_array_device, memory_for_population);

		/* allocate and zeroise scores array, avoid any future issues with non initialized arrays */
		long *scores_array_host = new long[number_individuals];
		long *scores_array_device;
		for(int contador=0;contador<number_individuals;contador++){
			scores_array_host[contador] = 0L;
		}
		size_t memory_for_scores = number_individuals*sizeof(long);
		cudaMalloc((void **) &scores_array_device, memory_for_scores);

		/*  allocate and initialize memory for acceptable result flag, flag indicates the element of the population which has the result */
		int *solution_found_host = new int;
		*solution_found_host = -1;
		int *solution_found_device;
		size_t memory_solution_found = sizeof(int);
		cudaMalloc((void **) &solution_found_device, memory_solution_found);



		/* we move data from host to device*/
		cudaMemcpy(population_array_device, population_array_host, memory_for_population, cudaMemcpyHostToDevice);
		cudaMemcpy(scores_array_device, scores_array_host, memory_for_scores, cudaMemcpyHostToDevice);
		cudaMemcpy(solution_found_device, solution_found_host, memory_solution_found, cudaMemcpyHostToDevice);


		/* we calculate dimensions for grid and blocks and create them: for evaluation */
		unsigned int blocks_required_evaluation = number_individuals/(threads_per_block_evaluation *individuals_per_thread_evaluation) + 
																	 (number_individuals%(threads_per_block_evaluation *individuals_per_thread_evaluation) == 0 ? 0:1);

		dim3 grid_evaluation(1,blocks_required_evaluation); /* in terms of blocks */
		dim3 block_evaluation(1,threads_per_block_evaluation); /* in terms of threads*/


		/*  define how many elements per thread, threads and blocks should be launched to scan the score of each individual, we create dim elements accordingly*/

		const unsigned int individuals_per_thread_scan_scores = 50;
		const unsigned int threads_per_block_scan_scores = 511; // remember block is 1 thread width and threads_per_block_scan_scores height
		
		const unsigned int blocks_required_scan_scores = (number_individuals/ (individuals_per_thread_scan_scores * threads_per_block_scan_scores)) +
														  (number_individuals%(threads_per_block_scan_scores * individuals_per_thread_scan_scores) == 0 ? 0:1);

		dim3 grid_scan_scores(1,blocks_required_scan_scores); // in terms of blocks
		dim3 block_scan_scores(1,threads_per_block_scan_scores); // in terms of threads



		/* output parameters */

		cout << "-Algorithm parameters-" << endl;
		cout << "Individuals: " << number_individuals << endl;
		cout << "Genes per individual: " << number_genes << endl;
		cout << "Individuals computed per thread: " << individuals_per_thread_evaluation << endl;
		cout << "-Computing distribution for evaluation-" << endl;
		cout << "Blocks required: " << blocks_required_evaluation << endl;
		cout << "Threads per block: " << threads_per_block_evaluation << endl;
		cout << "Total number of threads: " << blocks_required_evaluation*threads_per_block_evaluation << endl;
		cout << "-Computing distribution for scan_results-" << endl;
		cout << "Individuals (scores) evaluated per thread: " << individuals_per_thread_scan_scores << endl;
		cout << "Threads per block: " << threads_per_block_scan_scores << endl;
		cout << "Blocks required: " << blocks_required_scan_scores << endl;

		cout << endl << "Algorithm Start" << endl;



		/*we launch evaluation kernel: evaluate(int *input, int totalSizeOfArray, int number_genes, int individualsPerThread, int number_blocks, int threads_per_block, long *scores)*/
		
		
		evaluate <<< grid_evaluation, block_evaluation >>> (population_array_device, number_genes*number_individuals, number_genes, individuals_per_thread_evaluation, blocks_required_evaluation, threads_per_block_evaluation, scores_array_device);
																/* long *scores_array, int number_individuals, int individuals_per_thread, int threads_per_block, int *solution_found_flag, unsigned long desired_number, int acceptable_error */
		scan_for_solution <<< grid_scan_scores, block_scan_scores >>> (scores_array_device, number_individuals, individuals_per_thread_scan_scores, threads_per_block_scan_scores, solution_found_device, desired_number, acceptable_error_window);
		
    

		//cudaMemcpy(scores_array_host, scores_array_device, memory_for_scores, cudaMemcpyDeviceToHost);
		//cudaMemcpy(population_array_host, population_array_device, memory_for_population, cudaMemcpyDeviceToHost);
		cudaMemcpy(solution_found_host, solution_found_device, memory_solution_found, cudaMemcpyDeviceToHost);
		cout << *solution_found_host << endl;
		//cout << scores_array_host[*solution_found_host] << endl;

		
		return 0;
	}
