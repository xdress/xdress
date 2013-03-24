// Header for general library file.

#if !defined(_BRIGHT_)
#define _BRIGHT_

//standard libraries
#include <string>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <exception>
#include <sys/stat.h> 
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <typeinfo>

namespace bright {

  //Bright Globals
  void bright_start ();

  extern std::string BRIGHT_DATA;

  extern std::set<int> track_nucs;          // Set of isotopes to track for all components.
  extern std::vector<int> track_nucs_order; // Vector of isotopes to track for all components.

  extern void load_track_nucs_hdf5(std::string, std::string = "", bool = false);  //Load isotopic tracking list from HDF5 file.
  extern void load_track_nucs_text(std::string, bool = false);                    //Load isotopic tracking list from text file.

  extern void sort_track_nucs(); // Sets the isotopic tracking by zzaaam from lowest to highest and stores it in track_nucs_order

  extern int verbosity;			//How much should the components talk to us? 0 = None, 1 = a little, 2 = a lot!, etc.
  extern int write_text;
  extern int write_hdf5;

  extern std::string output_filename;

  // Some useful typedefs...
  typedef std::set<int> nuc_set;
  typedef nuc_set::iterator nuc_iter;

  // Fission Product HDF5 interface
  typedef struct fission_struct {
    char nuc_name[6];
    int nuc_zz;

    int8_t thermal_yield;
    int8_t fast_yield;
    int8_t high_energy_yield;

    double xs [63];
  } fission_struct;

  typedef struct fission_product_yields_struct {
    int16_t index;

    char from_nuc_name[6];
    int from_nuc_zz;

    char to_nuc_name[6];
    int to_nuc_zz;

    double mass_frac;
  } fission_product_yields_struct;

  typedef struct xs_1g_struct {
    char nuc_name[6];
    int nuc_zz;

    double sigma_t;
    double sigma_s;
    double sigma_e;
    double sigma_i;
    double sigma_a;

    double sigma_gamma;
    double sigma_f;
    double sigma_alpha;
    double sigma_proton;
    double sigma_duet;
    double sigma_trit;
    double sigma_2n;
    double sigma_3n;
    double sigma_4n;
  } xs_1g_struct;


  // Vectorized Functions
  std::vector<double> delta_vector(double, std::vector<double>);
  std::vector<double> normalized_delta(double, std::vector<double>);

  bool sorted_index_comparator(std::pair<int, double>, std::pair<int, double>);
  std::vector<int> sorted_index(std::vector<double>);

  std::vector<double> y_x_factor_interpolation(double, std::vector<double>, std::vector<double>);

  std::vector< std::vector<double> > vector_outer_product(std::vector<double>, std::vector<double>);
  std::vector< std::vector<double> > matrix_inverse(std::vector< std::vector<double> >);
  std::vector< std::vector<double> > matrix_addition(std::vector< std::vector<double> >, std::vector< std::vector<double> >);
  std::vector< std::vector<double> > matrix_multiplication(std::vector< std::vector<double> >, std::vector< std::vector<double> >);

  std::vector< std::vector<double> > scalar_matrix_product(double, std::vector< std::vector<double> >);
  std::vector<double> scalar_matrix_vector_product(double, std::vector< std::vector<double> >, std::vector<double>);

  //Array Methods
  template <class T>
  int find_index(T val, T * arr, int arr_len = -1)
  {
    //Finds an element 'val' in array 'arr'
    //returns the index of val's first location
    //returns -1 if not found.

    if (arr_len < 0)
      arr_len = sizeof(arr) / sizeof(T);

    for (int n = 0; n < arr_len; n++)
    {
      if (val == arr[n])
        return n;
    };

    return -1;
  };

  int find_index_char(char *, char **, int = -1);



  /******************/
  /*** Exceptions ***/
  /******************/

  class BadFuelForm : public std::exception
  {
  //Exception for valid fuel form.
  public:
    BadFuelForm () {};
    ~BadFuelForm () throw () {};

    static char * name ()
    {
      return (char *) "BadFuelForm";
    };

    virtual const char* what() const throw()
    {
      std::string BFFstr ("FUEL COMPOSITION NOT COMPUTABLE!");
      return (const char *) BFFstr.c_str();
    };
  };





  class VectorSizeError : public std::exception
  {
  //Exception for valid fuel form.
  public:
    VectorSizeError () {};
    ~VectorSizeError () throw () {};

    static char * name ()
    {
      return (char *) "VectorSizeError";
    };

    virtual const char* what() const throw()
    {
      std::string VWSstr ("Vector is of the wrong size.");
      return (const char *) VWSstr.c_str();
    };
  };





  class BisectionMethodNotPerformed : public std::exception
  {
  //Exception for when the bisection method is not calculated.
  public:
    BisectionMethodNotPerformed ()
    {
      errstr = "Bisection method was not performed.";
    };
    BisectionMethodNotPerformed (std::string calctype)
    {
      errstr = "Bisection method durring " + calctype + " calculation was not performed.";
    };
    ~BisectionMethodNotPerformed () throw () {};

    static char * name ()
    {
      return (char *) "BisectionMethodNotPerformed";
    };

    virtual const char* what() const throw()
    {
      return (const char *) errstr.c_str();
    };
  private:
    std::string errstr;
  };



  /***************************/
  /*** Sparse Matrix Stuff ***/
  /***************************/

  template <class T>
  class sparse_matrix_entry
  {
  public:
    int row;
    int col;
    T val;

    sparse_matrix_entry() {};
    ~sparse_matrix_entry() {};

    sparse_matrix_entry(int i, int j, T v)
    {
      row = i;
      col = j;
      val = v;
    };
  };


  template <class T>
  bool cmp_by_row (sparse_matrix_entry<T> a,sparse_matrix_entry<T> b) 
  {
    if (a.row != b.row)
      return (a.row < b.row);
    else
      return (a.col < b.col);
  };


  template <class T>
  bool cmp_by_col (sparse_matrix_entry<T> a,sparse_matrix_entry<T> b) 
  {
    if (a.col != b.col)
      return (a.col < b.col);
    else
      return (a.row < b.row);
  };


  template<class InputIterator, class T>
  InputIterator find_row( InputIterator first, InputIterator last, const T& value )
  {
    for ( ;first!=last; first++) 
    {
      if ((*first).row == value) 
        break;

      if (value < (*first).row)
        first = last - 1;
    };

    return first;
  };



  template<class InputIterator, class T>
  InputIterator find_col( InputIterator first, InputIterator last, const T& value )
  {
    for ( ;first!=last; first++) 
    {
      if ((*first).col == value) 
        break;

      if (value < (*first).col) 
        first = last - 1 ;
    };

    return first;
  };



  template <class T>
  class SparseMatrix
  {
  public:
    int nrows, ncols;
    std::vector< sparse_matrix_entry<T> > sm;

    SparseMatrix(){};
    ~SparseMatrix(){};

    SparseMatrix(int N, int nr=0, int nc = 0)
    {
      nrows = nr;
      ncols = nc;

      sm = std::vector< sparse_matrix_entry <T> >();
      sm.reserve(N);
    };

    int size()
    {
      return sm.size();
    };




    std::vector< std::vector<T> > todense()
    {
      int n;
      int N = size();
      typename std::vector< std::vector<T> > M = std::vector< std::vector<T> > (nrows, std::vector<T> (ncols, 0.0));
      for (n = 0; n < N; n++)
        M[sm[n].row][sm[n].col] = sm[n].val;
      return M;
    };


    void push_back(int i, int j, T value)
    {
      sm.push_back(sparse_matrix_entry<T>(i, j, value));
    };


    T at(int i, int j)
    {
      typename std::vector< sparse_matrix_entry<T> >::iterator a_iter = find_row(sm.begin(), sm.end(), i);
      while (i == (*a_iter).row)
      {
        if (j == (*a_iter).col)
          return (*a_iter).val;

        a_iter++;
      };
      return 0.0;
    };


    void sort_by_row()
    {
      std::sort(sm.begin(), sm.end(), cmp_by_row<T>);
    };


    void sort_by_col()
    {
      std::sort(sm.begin(), sm.end(), cmp_by_col<T>);
    };


    void clean_up()
    {
      // First, get all of your ducks in a row
      sort_by_row();

      int n, N;
      N = sm.size();
      std::vector<int> bad_ind = std::vector<int>();

      // Calculate indices to remove
      for (n = N - 1; 0 <= n; n--)
      {
        if ((sm[n].row == sm[n-1].row) && (sm[n].col == sm[n-1].col))
          bad_ind.push_back(n);
        else if (sm[n].val == 0.0)
          bad_ind.push_back(n);
      }; 

      // remove the offending indices
      int p, P;
      P = bad_ind.size();
      for (p = 0; p < P; p++)
        sm.erase(sm.begin()+bad_ind[p]);

      // Save some space
      sm.resize(sm.size());
    };


    friend std::ostream& operator<< (std::ostream& out, SparseMatrix<T> & A) 
    {
      int n = 0;
      int N = A.size();
        
      out << "Sparse Matrix [" << A.nrows << ", " << A.ncols << "] (" << N << ")\n";
      for (n = 0; n < N; n++)
        out << "  (" << A.sm[n].row << ", " << A.sm[n].col << ") = " << A.sm[n].val << "\n";

      return out;
    };



    double norm()
    {
      // Calculates the Frobenius norm for the sparse matrix
      int n, N;
      N = sm.size();
      double frob = 0.0;

      for (n = 0; n < N; n++)
        frob += (sm[n].val * sm[n].val);

      frob = sqrt(frob);
      return frob;
    };


    double abs_max()
    {
      int n, N;
      N = sm.size();
      double m = 0.0;

      for (n = 0; n < N; n++)
        if (m < fabs(sm[n].val))
          m = fabs(sm[n].val);

      return m;
    };


    void prune(double precision = 1E-10)
    {
      int n, N;
      N = sm.size();
      double cutoff = precision * abs_max();

      for (n = 0; n < N; n++)
        if (fabs(sm[n].val) < cutoff)
          sm[n].val = 0.0;

      clean_up();
    };


    void find_inf()
    {
      // Calculates the Frobenius norm for the sparse matrix
      int n, N;
      N = sm.size();
      double infin = 1.0 / 0.0;

      for (n = 0; n < N; n++)
        if (sm[n].val == infin)
          std::cout << "  (" << sm[n].row << ", " << sm[n].col << ") = " << sm[n].val << "\n";
    };




    SparseMatrix<T> transpose()
    {
      int n;
      int N = size();
      SparseMatrix<T> B = SparseMatrix<T>(N, nrows, ncols);

      for (n = 0; n < N; n++)
        B.push_back(sm[n].col, sm[n].row, sm[n].val);

      B.clean_up();
      return B;
    };



    SparseMatrix<T> operator* (double s)
    {
      int n;
      int N = size();
      SparseMatrix<T> B = SparseMatrix<T>(N, nrows, ncols);

      for (n = 0; n < N; n++)
        B.push_back(sm[n].row, sm[n].col, sm[n].val * s);

      B.clean_up();
      return B;
    };


    std::vector<double> operator* (std::vector<double> vec)
    {
      int n, i, j;
      int N = size();
      int P = vec.size();

      if (P != nrows && P != ncols)
        throw VectorSizeError();

      std::vector<double> new_vec = std::vector<double>(P, 0.0);

      for (n = 0; n < N; n++)
        new_vec[sm[n].row] += (sm[n].val * vec[sm[n].col]);

      return new_vec;
    };


    SparseMatrix<T> operator* (SparseMatrix<T> B)
    {
      int i, j;
      int N = size();

      if (B.nrows != nrows && B.ncols != ncols)
        throw VectorSizeError();

      // Put B in col-order
      B.sort_by_col();

      typename std::vector< sparse_matrix_entry<T> >::iterator a_iter, b_iter, a_stor, b_stor, a_beg, b_beg, a_end, b_end;
      a_beg = sm.begin();
      a_end = sm.end();
      a_stor = a_beg;

      b_beg = B.sm.begin();
      b_end = B.sm.end();
      b_stor = b_beg;

      SparseMatrix<T> C = SparseMatrix<T>(N, nrows, ncols);

      double dot_prod;
      for (i = 0; i < C.nrows; i++)
      {
        for (j = 0; j < C.ncols; j++)
        {
          a_iter = find_row(a_stor, a_end, i);
          b_iter = find_col(b_stor, b_end, j);

//        std::cout << "(" << i << ", " << j << ") = a(" << (*a_iter).row << ", " << (*a_iter).col << ") = b(" << (*b_iter).row << ", " << (*b_iter).col << ")\n";

          a_stor = a_iter;
          b_stor = b_iter;

          if ((a_iter == a_end) || (b_iter == b_end))
          {
            if (a_iter == a_end)
              a_stor = a_beg;

            if (b_iter == b_end)
              b_stor = b_beg;

            continue;
          };

          dot_prod = 0.0;

          while(((*a_iter).row == i) && ((*b_iter).col == j) && (a_iter != a_end) && (b_iter != b_end))
          {
            if ((*a_iter).col == (*b_iter).row)
            {
              dot_prod += ((*a_iter).val * (*b_iter).val);
              a_iter++;
              b_iter++;
            }
            else if ((*a_iter).col < (*b_iter).row)
              a_iter++;
            else if ((*b_iter).row < (*a_iter).col)
              b_iter++;
            else
              break;
            };

            // Add entry, if not sparse
            if (dot_prod != 0.0)
              C.push_back(i, j, dot_prod);

            if ((a_iter == a_beg) || (a_iter == a_end) || (a_iter == a_end - 1))
              a_stor = a_beg;

            if ((b_iter == b_beg) || (b_iter == b_end) || (b_iter == b_end - 1))
              b_stor = b_beg;
        };
      };

      // Put B back in the right order
      B.sort_by_row();

      C.clean_up();
      return C;
    };



    SparseMatrix<T> operator+ (SparseMatrix<T> B)
    {
      int i, j;
      int N = size();

      if (B.nrows != nrows && B.ncols != ncols)
        throw VectorSizeError();

      typename std::vector< sparse_matrix_entry<T> >::iterator a_iter, b_iter, a_end, b_end;
      a_end = sm.end();
      b_end = B.sm.end();

      SparseMatrix<T> C = SparseMatrix<T>(N + B.size(), nrows, ncols);

      double tmp_sum;
      for (i = 0; i < C.nrows; i++)
      {
        a_iter = find_row(sm.begin(), a_end, i);
        b_iter = find_row(B.sm.begin(), b_end, i);

        // Cover the case where there are no a- or b-entries for row == i
        if ((a_iter == a_end) && (b_iter == b_end))
          continue;

        // Cover the case where there are no b-entries for row == i
        if ((a_iter != a_end) && (b_iter == b_end))
        {
          while((*a_iter).row == i)
          {
              C.push_back(i, (*a_iter).col, (*a_iter).val);
              a_iter++;
          };
          continue;
        };

        // Cover the case where there are no a-entries for row == i
        if ((a_iter == a_end) && (b_iter != b_end))
        {
          while((*b_iter).row == i)
          {
            C.push_back(i, (*b_iter).col, (*b_iter).val);
            b_iter++;
          };
          continue;
        };

        // cover the case when there are both a and b entries for row == i
        while(((*a_iter).row == i) || ((*b_iter).row == i))
        {
          if (((*a_iter).row == i) && ((*b_iter).row == i))
          {
            if ((*a_iter).col == (*b_iter).col)
            {
              tmp_sum = ((*a_iter).val + (*b_iter).val);
              if (tmp_sum != 0.0)
                C.push_back(i, (*a_iter).col, tmp_sum);

              a_iter++;
              b_iter++;
            }
            else if ((*a_iter).col < (*b_iter).col)
            {
              C.push_back(i, (*a_iter).col, (*a_iter).val);
              a_iter++;
            }
            else
            {
              C.push_back(i, (*b_iter).col, (*b_iter).val);
              b_iter++;
            };
          }
          else if ((*a_iter).row == i)
          {
            C.push_back(i, (*a_iter).col, (*a_iter).val);
            a_iter++;
          }
          else if ((*b_iter).row == i)
          {
            C.push_back(i, (*b_iter).col, (*b_iter).val);
            b_iter++;
          }
          else
            break;
        };
      };

      C.clean_up();
      return C;
    };


    std::vector<double> exp (std::vector<double> vec, int max_iter = 10000, double epsilon = 1e-16)
    {
      int n, p;
      int P = vec.size();
      double ind_rel_err = 0.0;
      double max_rel_err = 1.0;

      if (P != nrows && P != ncols)
        throw VectorSizeError();

      // Init vectors
      std::vector<double> new_vec = vec;
      std::vector<double> new_vec_last = vec;

      std::vector<double> V = ((*this) * new_vec);
      for (p = 0; p < P; p++)
        new_vec[p] = new_vec[p] + V[p];

      n = 2;
      while((n < max_iter) && (epsilon < max_rel_err))
      {
        max_rel_err = 0.0;

        for (p = 0; p < P; p++)
          V[p] /= n;
        V = ((*this) * V);

        for (p = 0; p < P; p++)
        {
          new_vec[p] += V[p];

          // Calculate end contition
          ind_rel_err = fabs(1.0 - fabs(new_vec_last[p] / new_vec[p]));
          if (max_rel_err < ind_rel_err)
            max_rel_err = ind_rel_err;
        };

        std::cout << "  n = " << n << ", iso = " << new_vec[2812] << ", err = " << max_rel_err << "\n";

        new_vec_last = new_vec;
        n++;
      };

      std::cout << "  matrix exp found at iter = " << n << "\n";

      return new_vec;
    };
  };



// End bright namespace
};

#endif
