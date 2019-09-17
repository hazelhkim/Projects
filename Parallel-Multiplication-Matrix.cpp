#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;
int main(int argc, char *argv[])
{
    double begin, end, time_spent;
    int r, i, j, k, n, rowA, rowB, colA, colB;
    cout << "Enter number of rows and columns of matrix A, B : ";
    cin >> r;
    
    vector< vector<int> > A(r,vector<int>(r));
    vector< vector<int> > B(r,vector<int>(r));
    vector<int> C(r*r);
    vector<int> R(r*r);
    
    for (i = 0; i < r; i++)
        for (j = 0; j < r; j++)
            A[i][j] = rand()%10;
    
    for (i = 0; i < r; i++)
        for (j = 0; j < r; j++)
            B[i][j]= rand()%10;
    
    //test to see whether the for-loop works fine.
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < r; j++)
            cout << B[i][j] << " ";
        cout << "\n";
    }
    
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    
    // Get the number of processes
    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    
    // Get the rank of the process
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    
    // Print off a hello world message
    printf("Hello world from processor rank %d"
           " out of %d processors\n", myid, numprocs);
    
    if (myid == 0) {
        printf("Enter the number of buf: ");
        scanf("%d",&n);
        begin = MPI_Wtime();
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    for (i = 0 + myid; i < r*r; i+=numprocs)
    {
        colA = i % r;
        rowA = (i - colA) / r;
        rowB = i % n;
        colB = (i - rowB) / r;
        for (k = 0; k < r; i++) {
            C[rowA*r + colA] += A[k][colA] * B[rowB][k];
            printf("Hello world from processor rank %d"
               " out of %d processors\n", myid, numprocs);
        }
    }
    MPI_Reduce(&C, &R, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < r; j++)
            cout << B[i][j] << " ";
        cout << "\n";
    }
    
    if (myid == 0) {
        end = MPI_Wtime();
        time_spent = (double)(end - begin);
        cout << "Matrix Size :" << r << " = Time: "<<time_spent;
        cout << "\n";
    }
    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}


