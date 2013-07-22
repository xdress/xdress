// General Library 

#include "bright.h"

//Bright Globals

std::string bright::BRIGHT_DATA = "";

void bright::bright_start()
{
  #ifdef _WIN32
    char * tmpBRIGHT_DATA;
    size_t lenBRIGHT_DATA;
    errno_t errBRIGHT_DATA = _dupenv_s(&tmpBRIGHT_DATA, &lenBRIGHT_DATA, "BRIGHT_DATA");
    if (errBRIGHT_DATA) std::cout << "BRIGHT_DATA Enviromental Variable could not be found\n";
      BRIGHT_DATA = (std::string) tmpBRIGHT_DATA;
  #else
    BRIGHT_DATA = getenv("BRIGHT_DATA");
  #endif
  return;
};



#ifdef _WIN32
  int null_set [1] = {922350};
  std::set<int> bright::track_nucs (null_set, null_set+1);
  std::vector<int> bright::track_nucs_order (null_set, null_set+1);
#else
  int null_set [0] = {};
  std::set<int> bright::track_nucs (null_set, null_set+0);
  std::vector<int> bright::track_nucs_order (null_set, null_set+0);
#endif

int bright::verbosity  = 0;
int bright::write_text = 1;
int bright::write_hdf5 = 0;

std::string bright::output_filename = "fuel_cycle.h5";


void bright::sort_track_nucs()
{
  track_nucs_order = std::vector<int> (track_nucs.begin(), track_nucs.end());
  std::sort(track_nucs_order.begin(), track_nucs_order.end());
};



void bright::load_track_nucs_hdf5(std::string filename, std::string datasetname, bool clear_prev)
{
};

void bright::load_track_nucs_text(std::string filename, bool clear_prev)
{
};




/*
 *  Vectorized functions
 */

std::vector<double> bright::delta_vector(double x, std::vector<double> vec)
{
  // This functions finds the 
  // value of (x - vec[i]) for all elements i 
  // in the vector.
  std::vector<double> d (vec.size(), 0.0);

  // Calculate the normalized delta for 
  // all i elements.
  for(int i = 0; i < vec.size(); i++)
    d[i] = (x - vec[i]);

  return d;
};





std::vector<double> bright::normalized_delta(double x, std::vector<double> vec)
{
  // This functions find the normalized 
  // value of (x - vec[i]) for all elements i 
  // in the vector.
  //
  // This is equivelent to the fraction:
  //     (x - vec[i])
  //     ------------
  //      norm_factor
  //
  // Where the normalization factor is 
  //   norm_factor = (vec_max - vec_min) 
  // if the min does not equal the max.
  // and norm_factor = vec_min = vec_max 
  // if it does.

  double norm_factor;
  std::vector<double> nd (vec.size(), 0.0);

  // Get the min and max out of the vector
  double vec_min = *std::min_element(vec.begin(), vec.end());
  double vec_max = *std::max_element(vec.begin(), vec.end());

  if (vec_min == vec_max)
    norm_factor = vec_min;
  else
    norm_factor = vec_max - vec_min;

  // Calculate the normalized delta for 
  // all i elements.
  for(int i = 0; i < vec.size(); i++)
    nd[i] = (x - vec[i]) / norm_factor;

  return nd;
};



bool bright::sorted_index_comparator(std::pair<int, double> i, std::pair<int, double> j)
{
  return i.second < j.second;
};


std::vector<int> bright::sorted_index(std::vector<double> vec)
{
  // Make an indexed vector
  int I = vec.size();
  std::vector< std::pair<int, double> > ind_vec (I);
  for (int i = 0; i < I; i++)
    ind_vec[i] = std::pair<int, double>(i, vec[i]);

  // Sort the indexed vector
  std::sort(ind_vec.begin(), ind_vec.end(), sorted_index_comparator);

  // Grab the indicies out of ind_vec
  std::vector<int> ind (I);
  for (int i = 0; i < I; i++)
    ind[i] = ind_vec[i].first;

  return ind;
};




std::vector<double> bright::y_x_factor_interpolation(double x_factor, std::vector<double> y2, std::vector<double> y1)
{
  // This function calculates the following equation in a vectorized way
  //
  //      y = x(y2 - y1) * x_factor + y1
  //
  // y1 must be of the same size as y2

  int N = y1.size();

  std::vector<double> y (N, -1.0);

  for (int n = 0; n < N; n++)
    y[n] = ((y2[n] - y1[n]) * x_factor) + y1[n];

  return y;
};






std::vector< std::vector<double> > bright::vector_outer_product(std::vector<double> a, std::vector<double> b)
{
  // Performs outer product operation on two vectors
  int I = a.size(); 

  if (I != b.size())
    throw VectorSizeError();

  std::vector< std::vector<double> > c (I, std::vector<double>(I, 0.0)); 

  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < I; j++)
      c[i][j] = a[i] * b[j];
  };

  return c;
};






std::vector< std::vector<double> > bright::matrix_inverse(std::vector< std::vector<double> > a)
{
  // Performs outer product operation on two vectors
  int I = a.size(); 

  std::vector< std::vector<double> > a_inv (I, std::vector<double>(I, 0.0)); 

  /* This function calculates the inverse of a square matrix
   *
   * Code is rewritten from c++ template code Mike Dinolfo
   * by D. Kroon which was rewritten by Anthony Scopatz
   * which was found at http://snippets.dzone.com/posts/show/7558
   *
   */
  /* Loop variables */
  int i, j, k;

  /* Sum variables */
  double sum, x;
    
  /*  Copy the input matrix to output matrix */
  for (i = 0; i < I; i++) 
  {
    for (j = 0; j < I; j++)
      a_inv[i][j] = a[i][j]; 
  };
    
  /* Add small value to diagonal if diagonal is zero */
  for(i = 0; i < I; i++)
  { 
    if((a_inv[i][i] < 1e-12) && (a_inv[i][i] > -1e-12))
      a_inv[i][i] = 1e-12; 
  }
    
  /* Matrix size of one is special cased */
  if (I == 1)
  {
    a_inv[0][0] = 1.0 / a_inv[0][0];
    return a_inv;
  };

  /* Matrix size must be larger than zero */
  if (I <= 0)
    throw VectorSizeError();

  /* normalize row 0 */
  for (i = 1; i < I; i++) 
    a_inv[0][i] /= a_inv[0][0];

  /* Do LU separation */    
  for (i = 1; i < I; i++)  
  {
    /* do a column of L */
    for (j = i; j < I; j++)  
    { 
      sum = 0.0;
      for (k = 0; k < i; k++) 
        sum += a_inv[j][k] * a_inv[k][i];

      a_inv[j][i] -= sum;
    };

    if (i == I-1)
      continue;

        
    /* do a row of U */
    for (j = i+1; j < I; j++)
    {
      sum = 0.0;
      for (k = 0; k < i; k++)
        sum += a_inv[i][k] * a_inv[k][j];

      a_inv[i][j] = (a_inv[i][j] - sum) / a_inv[i][i];
    };
  };

    /* invert L */ 
    for ( i = 0; i < I; i++ )  
    {
        for ( j = i; j < I; j++ )  
        {
            x = 1.0;

            if ( i != j ) 
            {
                x = 0.0;
                for ( k = i; k < j; k++ ) 
                    x -= a_inv[j][k] * a_inv[k][i];
            };

            a_inv[j][i] = x / a_inv[j][j];
        };
    };

  /* invert U */ 
  for ( i = 0; i < I; i++ ) 
  {
    for ( j = i; j < I; j++ )  
    {
      if ( i == j ) 
        continue;

      sum = 0.0;
      for ( k = i; k < j; k++ )
        sum += a_inv[k][j] * ( (i==k) ? 1.0 : a_inv[i][k] );

      a_inv[i][j] = -sum;
    };
  };

  /* final inversion */ 
  for ( i = 0; i < I; i++ ) 
  {
    for ( j = 0; j < I; j++ )  
    {
      sum = 0.0;

      for ( k = ((i>j)?i:j); k < I; k++ ) 
        sum += ((j==k)?1.0:a_inv[j][k]) * a_inv[k][i];

      a_inv[j][i] = sum;
    };
  };
 
  return a_inv;
};





std::vector< std::vector<double> > bright::matrix_addition(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b)
{
  // Adds two matrices together

  int I = a.size();

  if ( I != a[0].size() || I != b.size() || I != b[0].size())
    throw VectorSizeError();
    
  std::vector< std::vector<double> > c (I, std::vector<double>(I, 0.0)); 

  int i, j;

  for (i = 0; i < I; i++)
  {
    for (j = 0; j < I; j++)
      c[i][j] = a[i][j] + b[i][j];
  };

  return c;
};




std::vector< std::vector<double> > bright::matrix_multiplication(std::vector< std::vector<double> > a, std::vector< std::vector<double> > b)
{
  // Multiplies two matrices together

  int I = a.size();

  if ( I != a[0].size() || I != b.size() || I != b[0].size())
    throw VectorSizeError();
    
  std::vector< std::vector<double> > c (I, std::vector<double>(I, 0.0)); 

  int i, j, k;

  for (i = 0; i < I; i++)
  {
    for (j = 0; j < I; j++)
    {
      for (k = 0; k < I; k++)        
        c[i][j] += a[i][k] * b[k][j];
    };
  };

  return c;
};





std::vector< std::vector<double> > bright::scalar_matrix_product(double a, std::vector< std::vector<double> > M)
{
  // Solves the equation r = aM for a scalar a and Matrix M.
  // Returns the resultant vector r.

  int I = M.size();

  if (I != M[0].size())
    throw VectorSizeError();

  std::vector< std::vector<double> > r (I, std::vector<double>(I, 0.0)); 

  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < I; j++)
      r[i][j] += (a * M[i][j]);
  };

  return r;
};





std::vector<double> bright::scalar_matrix_vector_product(double a, std::vector< std::vector<double> > M, std::vector<double> v)
{
  // Solves the equation r = aMv for a scalar a, Matrix M, and vector v.
  // Returns the resultant vector r.

  int I = M.size();

  if ( I != M[0].size() || I != v.size())
    throw VectorSizeError();

  std::vector<double> r (I, 0.0);

  for (int i = 0; i < I; i++)
  {
    for (int j = 0; j < I; j++)
      r[i] += (M[i][j] * v[j]);

    r[i] = (r[i] * a);
  };

  return r;
};





/* 
 * Array Helpers
 */

int bright::find_index_char(char * val, char ** arr, int arr_len)
{
  // Finds an element 'val' in array 'arr'
  // returns the index of val's first location
  // returns -1 if not found.
  // For Arrays of char strings

  if (arr_len < 0)
    arr_len = 1;

  for (int n = 0; n < arr_len; n++)
  {
    if (strcmp(arr[n], val) == 0)
       return n;
  };

  return -1;
};

