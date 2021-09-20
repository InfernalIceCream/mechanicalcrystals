#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <lapacke.h>

/* Adapted/taken from the time independent heat equation code from lecture 9 */

struct band_mat{
	long ncol;        /* Number of columns in band matrix           */
	long nbrows;      /* Number of rows (bands in original matrix)  */
	long nbands_up;   /* Number of bands above diagonal             */
	long nbands_low;  /* Number of bands below diagonal             */
	double *array;    /* Storage for the matrix in banded format    */
	// Internal temporary storage for solving inverse problem
	long nbrows_inv;  /* Number of rows of inverse matrix   		*/
	double *array_inv;/* Store the inverse if this is generated 	*/
	int *ipiv;        /* Additional inverse information         	*/
};
typedef struct band_mat band_mat;

/* Initialise a band matrix of a certain size, allocate memory,
   and set the parameters.  */
int init_band_mat(band_mat *bmat, long nbands_lower, long nbands_upper, long n_columns) {
	bmat->nbrows     = nbands_lower + nbands_upper + 1;
	bmat->ncol       = n_columns;
	bmat->nbands_up  = nbands_upper;
	bmat->nbands_low = nbands_lower;
	bmat->array      = (double *) malloc(sizeof(double)*bmat->nbrows*bmat->ncol);
	bmat->nbrows_inv = bmat->nbands_up*2 + bmat->nbands_low + 1;
	bmat->array_inv  = (double *) malloc(sizeof(double)*(bmat->nbrows+bmat->nbands_low)*bmat->ncol);
	bmat->ipiv       = (int *) malloc(sizeof(int)*bmat->ncol);
	if (bmat->array == NULL || bmat->array_inv == NULL) {
		return 0;
	}
	/* Initialise array to zero */
	long i;
	for (i = 0; i < bmat->nbrows * bmat->ncol; i++) {
		bmat->array[i] = 0.0;
	}
	return 1;
};

/* Get a pointer to a location in the band matrix, using
   the row and column indexes of the full matrix.           */
double *getp(band_mat *bmat, long row, long column) {
	int bandno = bmat->nbands_up + row - column;
	if (row < 0 || column < 0 || row >= bmat->ncol || column >= bmat->ncol) {
		printf("Indexes out of bounds in getp: %ld %ld %ld \n",row,column,bmat->ncol);
		exit(1);
	}
	return &bmat->array[bmat->nbrows*column + bandno];
}

/* Return the value of a location in the band matrix, using
   the row and column indexes of the full matrix.           */
double getv(band_mat *bmat, long row, long column) {
	return *getp(bmat,row,column);
}

double setv(band_mat *bmat, long row, long column, double val) {
	*getp(bmat,row,column) = val;
	return val;
}

/* Solve the equation Ax = b for a matrix a stored in band format
   and x and b real arrays                                          */
int solve_Ax_eq_b(band_mat *bmat, double *x, double *b) {
	/* Copy bmat array into the temporary store */
	int i,bandno;
	for (i = 0; i < bmat->ncol; i++) { 
		for (bandno = 0; bandno < bmat->nbrows; bandno++) {
			bmat->array_inv[bmat->nbrows_inv*i+(bandno+bmat->nbands_low)] = bmat->array[bmat->nbrows*i+bandno];
		}
		x[i] = b[i];
	}
	
	long nrhs = 1;
	long ldab = bmat->nbands_low*2 + bmat->nbands_up + 1;
	int info = LAPACKE_dgbsv( LAPACK_COL_MAJOR, bmat->ncol, bmat->nbands_low, bmat->nbands_up, nrhs, bmat->array_inv, ldab, bmat->ipiv, x, bmat->ncol);
	return info;
}

/* Reindexing function based on lecture */
long reindex(long index, long ncols) {
	long k;
	
	if (index < (ncols/2)) {
		k = 2*index;
	} else {
		long j = 2*(ncols-index)-1;
		k = j;
	}
	
	return k;
}

// Represents matrix in a non-band format
int printmat(band_mat *bmat) {
	long ncols = bmat->ncol;
	
	for (long i = 0; i < ncols; i++) {
		for (long j = 0; j < ncols; j++) {
			if (getp(bmat, reindex(i, ncols), reindex(j, ncols)) == NULL) {
				printf(" %0.2lf ", 0.0);
			} else {
				printf(" %0.2lf ", getv(bmat,reindex(i, ncols), reindex(j, ncols)));
			}
		}
		printf("\n");
	}
	
	printf("\n");
	return 0;
}

// Freeing band matrix memory function
void free_band_mat(band_mat *bmat) {
	free(bmat->array);
	free(bmat->array_inv);
	free(bmat->ipiv);
}

// Declaring read_input function before main
void read_input(double *L, int *N, double *K_p, double *gamma_p, double *w_zero_p, double *w_one_p, int *J_p);

// Diagnostic routines: set to 1 to enable.
#define DIAGS 0

// |***************|
// | Main Function |
// |***************|

int main(void) {
	long i; // Looping variable
	
	// **********
	// Parameters
	// **********

	// Right x boundary of domain
	double L;
	double *L_p = &L;
	// Number of grid points
	int N;
	int *N_p = &N;
	// Wavenumber of forcing
	double K;
	double *K_p = &K;
	// Damping rate
	double gamma;
	double *gamma_p = &gamma;
	// Minimum frequency in frequency scan
	double w_zero;
	double *w_zero_p = &w_zero;
	// Maximum frequency in frequency scan
	double w_one;
	double *w_one_p = &w_one;
	// Number of values in the scan over w
	int J;
	int *J_p = &J;
	
	// Read in from file; 
	read_input(L_p, N_p, K_p, gamma_p, w_zero_p, w_one_p, J_p);
	
	/* Mu function (mass density) and E function 
	   (elasticity) (taken in as a set of values) */
	double *Mu_f, *E_f;
	Mu_f = (double*) malloc(N*sizeof(double));
	E_f  = (double*) malloc(N*sizeof(double));
	
	// Read in values of Mu_f and K_f at grid points
	FILE *coeff;
	if(!(coeff=fopen("coefficients.txt","r"))) {
		printf("Error opening file\n");
		exit(1);
	}
	for (i = 0; i < N; i++) {
		if(2!=fscanf(coeff,"%lf %lf",&Mu_f[i],&E_f[i])) {
			printf("Error reading parameters from file\n");
			exit(1);
		}
	}
	fclose(coeff);
	
	FILE *file;
	file = fopen("output.txt","w");
	if (file == NULL) {
		return 1;
	}
	
	band_mat bmat;
	long ncols = 2*N;		// number of columns in matrix
	double dx = L/N;	// spacial step. L/N to exclude gridpoint x = L
	
	long nbands_low = 4;  
	long nbands_up  = 4;
	init_band_mat(&bmat, nbands_low, nbands_up, ncols);
	
	double *x = malloc(sizeof(double)*ncols);		// output vector for h(x)
	double *b = malloc(sizeof(double)*ncols);		// source vector b in matrix form Ax = b
	
	long d = 0;		// 2nd counting variable for index of real/imag components for setting b in Ax = b
					// (getting x coord d*dx)
	
	for (i = 0; i < ncols; i = i+2) {
		b[reindex(i,ncols)] = -cos(K*d*dx);
		b[reindex(i+1,ncols)] = -sin(K*d*dx);
		d++;
	}
	
	/*  Loop over the equation number and set the matrix
		values equal to the coefficients of the grid values 
		note boundaries treated with special cases       */
		
	for (int j = 0; j < J; j++) {
		
		long c = 1;		// counting variable for index of real/imag components a_c + ib_c
		
		double w_j = w_zero + (j*(w_one - w_zero))/J;	// Used frequency on loop
		double Q = w_j*gamma*dx*dx;
		
		double R_boundtop = w_j*w_j*dx*dx*Mu_f[0] - E_f[1] - E_f[0];	// coefficient of main diagonal for top boundary
		
		/* Boundary condition for a_0 + ib_0. Top left corner of matrix part */
		// Coeffs for real
		setv(&bmat, reindex(0, ncols), reindex(0, ncols), R_boundtop/(dx*dx));		// a_0
		setv(&bmat, reindex(0, ncols), reindex(1, ncols), 		   Q/(dx*dx));		// b_0
		setv(&bmat, reindex(0, ncols), reindex(2, ncols),  	  E_f[1]/(dx*dx));		// a_1
		
		// Coeffs for imag
		setv(&bmat, reindex(1, ncols), reindex(0, ncols), 		  -Q/(dx*dx));		// a_0
		setv(&bmat, reindex(1, ncols), reindex(1, ncols), R_boundtop/(dx*dx));		// b_0
		setv(&bmat, reindex(1, ncols), reindex(2, ncols), 				   0);		// a_1
		setv(&bmat, reindex(1, ncols), reindex(3, ncols),  	  E_f[1]/(dx*dx));		// b_1
		
		/* Boundary condition for a_0 + ib_0. Top right corner of matrix part */
		// Coeffs for real
		setv(&bmat, reindex(0, ncols), reindex(ncols-2, ncols),  E_f[0]*cos(K*L)/(dx*dx));		// a_N-1
		setv(&bmat, reindex(0, ncols), reindex(ncols-1, ncols),  E_f[0]*sin(K*L)/(dx*dx));		// b_N-1
		
		// Coeffs for imag
		setv(&bmat, reindex(1, ncols), reindex(ncols-2, ncols), -E_f[0]*sin(K*L)/(dx*dx));		// a_N-1
		setv(&bmat, reindex(1, ncols), reindex(ncols-1, ncols),  E_f[0]*cos(K*L)/(dx*dx));		// b_N-1
		  
		for (i = 2; i < ncols-2; i = i+2) {
			
			double R = w_j*w_j*dx*dx*Mu_f[c] - E_f[c+1] - E_f[c];	// coefficient of main diagonal
			
			// Coeffs for real a_c (even number row)
			setv(&bmat,	  reindex(i, ncols),  reindex(i-2, ncols),   E_f[c]/(dx*dx));		// a_c-1
			setv(&bmat,	  reindex(i, ncols),  reindex(i-1, ncols),	       	      0);		// b_c-1
			setv(&bmat,	  reindex(i, ncols),  reindex(  i, ncols),	      R/(dx*dx));		// a_c
			setv(&bmat,	  reindex(i, ncols),  reindex(i+1, ncols),        Q/(dx*dx));		// b_c
			setv(&bmat,	  reindex(i, ncols),  reindex(i+2, ncols), E_f[c+1]/(dx*dx));		// a_c+1
			
			// Coeffs for imag b_c (odd number row)
			setv(&bmat,	reindex(i+1, ncols),  reindex(i-1, ncols),   E_f[c]/(dx*dx));		// b_c-1
			setv(&bmat,	reindex(i+1, ncols),  reindex(  i, ncols),	     -Q/(dx*dx));		// a_c
			setv(&bmat,	reindex(i+1, ncols),  reindex(i+1, ncols),		  R/(dx*dx));		// b_c
			setv(&bmat,	reindex(i+1, ncols),  reindex(i+2, ncols),		  		  0);		// a_c+1
			setv(&bmat,	reindex(i+1, ncols),  reindex(i+3, ncols), E_f[c+1]/(dx*dx));		// b_c+1
			
			c++;	// increment index of real/imag components
		}
		
		double R_boundbot = w_j*w_j*dx*dx*Mu_f[N-1] - E_f[0] - E_f[N-1];	// coefficient of main diagonal for bot boundary
		
		/* Boundary condition for a_N-1 + ib_N-1. Bottom right corner of the matrix part */
		// Coeffs for real
		setv(&bmat, reindex(ncols-2, ncols), reindex(ncols-4, ncols),    E_f[N-1]/(dx*dx));		// a_N-2
		setv(&bmat, reindex(ncols-2, ncols), reindex(ncols-3, ncols),  					0);		// b_N-2
		setv(&bmat, reindex(ncols-2, ncols), reindex(ncols-2, ncols),  R_boundbot/(dx*dx));		// a_N-1
		setv(&bmat, reindex(ncols-2, ncols), reindex(ncols-1, ncols), 		   -Q/(dx*dx));		// b_N-1
		
		// Coeffs for imag
		setv(&bmat, reindex(ncols-1, ncols), reindex(ncols-3, ncols), 	E_f[N-1]/(dx*dx));		// b_N-2
		setv(&bmat, reindex(ncols-1, ncols), reindex(ncols-2, ncols), 		   Q/(dx*dx));		// a_N-1
		setv(&bmat, reindex(ncols-1, ncols), reindex(ncols-1, ncols), R_boundbot/(dx*dx));		// b_N-1
		
		/* Boundary condition for a_N-1 + ib_N-1. Bottom left corner of the matrix part */
		// Coeffs for real
		setv(&bmat, reindex(ncols-2, ncols), reindex(0, ncols),  E_f[0]*cos(K*L)/(dx*dx));		// a_0
		setv(&bmat, reindex(ncols-2, ncols), reindex(1, ncols), -E_f[0]*sin(K*L)/(dx*dx));		// b_0
		
		// Coeffs for imag
		setv(&bmat, reindex(ncols-1, ncols), reindex(0, ncols),  E_f[0]*sin(K*L)/(dx*dx));		// a_0
		setv(&bmat, reindex(ncols-1, ncols), reindex(1, ncols),  E_f[0]*cos(K*L)/(dx*dx));		// b_0
		
		/*  Print matrix for debugging: */ 
		if (DIAGS) {
			printmat(&bmat);            
		}
		
		solve_Ax_eq_b(&bmat,x,b);
		
		d = 0;			// Reusing d as a counting variable again for file output
		
		for (i = 0; i < ncols; i = i+2) {
			
			/* Prints to file in 4 column format (w_j, x,  x[i] real, x[i] imag) */
			fprintf(file, "%g %g %g %g \n",w_j,d*dx,x[reindex(i, ncols)],x[reindex(i+1, ncols)]);

			d++;
		}
	}
	fclose(file);
	
	free(Mu_f);
	free(E_f);
	free(x);
	free(b);
	free_band_mat(&bmat);
	return 0;
}

// Reading input from input.txt file. Adapted from assignment 3 code.
void read_input(double *L, int *N, double *K, double *gamma, double *w_zero, double *w_one, int *J) {
	FILE *infile;
		if(!(infile=fopen("input.txt","r"))) {
			printf("Error opening file\n");
			exit(1);
		}
		if(7!=fscanf(infile,"%lf %d %lf %lf %lf %lf %d",L,N,K,gamma,w_zero,w_one,J)) {
			printf("Error reading parameters from file\n");
			exit(1);
		}
	fclose(infile);
}
