#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <cuda.h>
using namespace std;

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

	/*gloabl position in population array index calculation */
	int startingPosition = (blockIdx.y * threads_per_block * number_genes) + (threadIdx.y * number_genes);
	if(startingPosition>=totalSizeOfArray){
		//return; //*return if thread is useless */
	}

	/*gloabl position in scores array index calculation  REDO*/
	int startingPosition_scores = (blockIdx.y * threads_per_block) + threadIdx.y;


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



	int main(){
		//define settings
		const unsigned int number_genes = 10;
		const unsigned int number_individuals = 101;

		const unsigned int threads_per_block_evaluation = 10; //DO NOT FORGET: BLOCK IS 1 thread width, and threads_per_block  height
		const unsigned int individuals_per_thread_evaluation = 3;


		//allocate and randomly initialize memory for population
		int *population_array_host = new int[number_genes*number_individuals];
		int *population_array_device;
		srand ( time(NULL) );
		for(int contador=0;contador<number_genes*number_individuals;contador++){
			population_array_host[contador] = ( rand()  % 10 );
		}
		size_t memory_for_population = number_genes*number_individuals*sizeof(int);
		cudaMalloc((void **) &population_array_device, memory_for_population);

		//allocate and zeroise scores array, avoid any future issues with non initialized arrays
		long *scores_array_host = new long[number_individuals];
		long *scores_array_device;
		for(int contador=0;contador<number_individuals;contador++){
			scores_array_host[contador] = 0L;
		}
		size_t memory_for_scores = number_individuals*sizeof(long);
		cudaMalloc((void **) &scores_array_device, memory_for_scores);

		//we move data from host to device
		cudaMemcpy(population_array_device, population_array_host, memory_for_population, cudaMemcpyHostToDevice);
		cudaMemcpy(scores_array_device, scores_array_host, memory_for_scores, cudaMemcpyHostToDevice);

		//we calculate dimensions for grid and blocks and create them: for evaluation
		unsigned int blocks_required_evaluation = number_individuals/(threads_per_block_evaluation *individuals_per_thread_evaluation) + 
																	 (number_individuals%(threads_per_block_evaluation *individuals_per_thread_evaluation) == 0 ? 0:1);

		dim3 grid_evaluation(1,blocks_required_evaluation); //in terms of blocks
		dim3 block_evaluation(1,threads_per_block_evaluation); // in terms of threads

		cout << "Blocks " << blocks_required_evaluation<< endl;
		cout << "Threads " << threads_per_block_evaluation<< endl;
		
		//we launch evaluation kernel: evaluate(int *input, int totalSizeOfArray, int number_genes, int individualsPerThread, int number_blocks, int threads_per_block, long *scores)
		evaluate <<< grid_evaluation, block_evaluation >>> (population_array_device, number_genes*number_individuals, number_genes, individuals_per_thread_evaluation, blocks_required_evaluation, threads_per_block_evaluation, scores_array_device);
		
		cudaMemcpy(scores_array_host, scores_array_device, memory_for_scores, cudaMemcpyDeviceToHost);
		cudaMemcpy(population_array_host, population_array_device, memory_for_population, cudaMemcpyDeviceToHost);


	for(int contador=0;contador<number_genes*number_individuals;contador++){
		if(contador%number_genes==0 && contador > 0){
			//cout << endl;
		}
		//cout << population_array_host[contador] << "-";

	}
	cout << endl;
	cout << "----";
	cout << endl;
	
	for(int contador=0;contador<number_individuals;contador++){
		cout << scores_array_host[contador] << endl;
	}

		return 0;
	}
