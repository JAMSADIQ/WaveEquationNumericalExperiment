/*A simple code for u_tt = c^2 u_xx with u[t=0,x] = sin(4pix/L), u_t[0,x]= 0 and u[t,x=0] = 0 =u[t, x=L]  x: [0,L=1], t: [0, T]  using first Explicit and than Implicit scheme*/

#include <iostream>
#include <math.h> //use for M_PI
using namespace std;

/*following book by Titus Beu */
double *Vector(int imin, int imax)
{
   double *p; // assign block start to array pointer
   p = (double*) malloc((size_t) ((imax-imin+1)*sizeof(double)));
   if (!p) {
      printf("Vector: allocation error !\n");
      exit(1);
   }
   return p - imin; //adjust for offset
}

void Explicitpropagte(double u0[], double u1[], double u[], int nx, double c, double dx, double dt)
//---------------------------------------------------------------------------
// Propagates the solutions u0[] and u1[] of the wave equation
//    u_tt = c^2 u_xx,  c=1.0 
// over the time interval dt, using the explicit difference scheme on a
// spatial grid with nx nodes and spacing dx. Returns the solution in u[].
//---------------------------------------------------------------------------
{
   double coeff, sqcoeff, coeff2;
   int i;

   coeff = c*dt/dx; 
   sqcoeff = coeff*coeff;
   coeff2 = 2e0*(1e0 - sqcoeff);
   /*I am following book so they use index 1 to N+1 not from 0 */
   u[1] = u0[1]; u[nx] = u0[nx];   // Dirichlet boundary conditions
   for (i=2; i<=nx-1; i++)         // propagate solution at interior points
      u[i] = sqcoeff*u1[i-1] + coeff2*u1[i] + sqcoeff*u1[i+1] - u0[i];
}

void Init(double u0[], double u1[], double x[], int nx, double c, double dx, double dt)
//---------------------------------------------------------------------------
// Returns initial solutions u0 and u1, for the first two time steps
//    u0(x,0) = sin(4 pi x) / (L = 1)                 initial solution
//    v0(x,0) = 0                                     initial time derivative
// x - spatial mesh, nx - number of nodes
// c -  velocity of wave,
// dx - x-spacing, dt - time step size
//---------------------------------------------------------------------------
{
   double coeff, coeff2, sqcoeff, v0;
   int i;
   for (i=1; i<=nx; i++)                             // time step 0
      u0[i] = sin(4*M_PI*x[i]/1.0); 

   coeff = c*dt/dx; sqcoeff = coeff*coeff;                  // time step 1
   coeff2 = 2e0*(1e0 - coeff);
   u1[1] = u0[1]; u1[nx] = u0[nx];                // constant boundary values
   for (i=2; i<=nx-1; i++) {
      v0 = 0e0;                                 // initial time derivative
      u1[i] = 0.5e0*(sqcoeff*u0[i-1] + coeff2*u0[i] + sqcoeff*u0[i+1]) - dt*v0;
   }
}
/*Explicit scheme and saving data*/
int main()
{
   double *u0, *u1, *u, *x;
   double c, dt, dx, t, tmax, xmax, cflcoeff;
   int i, it,  nout, nt, nx, nx2;
   char fname[80], title[80];
   FILE *out;

   c    = 1e0;                                        // phase speed of wave
   xmax = 1e0;                                                 // maximum x
   tmax = 40e0;                                   // maximum propagation time
   dt   = 2.5e-3;                                        // time step
   cflcoeff = 0.275;		             //CFL which we set
   dx   = c*dt/cflcoeff;                            // spatial step size
   nout = 500;
   nx = 2*(int)(xmax/dx + 0.5) + 1;            // odd number of spatial nodes
   nt = (int)(tmax/dt + 0.5);                         // number of time steps
   nx2 = nx/2;

   u0 = Vector(1,nx);                                     // initial solution
   u1 = Vector(1,nx);                            // first propagated solution
   u  = Vector(1,nx);                                             // solution
   x  = Vector(1,nx);                                         // spatial mesh

   for (i=1; i<=nx; i++) x[i] = (i-nx2-1)*dx;                 // spatial mesh

   Init(u0, u1, x, nx, c, dx, dt);
   for (it=1; it<=nt; it++) {                                    // time loop
      t = it*dt;
      Explicitpropagte(u0,u1,u,nx,c,dx,dt);                   // propagate solution
                                                           // shift solutions
      for (i=1; i<=nx; i++) { u0[i] = u1[i];  u1[i] = u[i]; }

      if (it % nout == 0 || it == nt) {            // output every nout steps
         sprintf(fname,"wave_%4.2f.txt",t);
         out = fopen(fname,"w");
         fprintf(out,"t = %4.2f\n",t);
         fprintf(out,"     x          u\n");
         for (i=1; i<=nx; i++)
            fprintf(out,"%10.5f%10.5f\n",x[i],u[i]);
         fclose(out);
      }
   }
}


/*Implicit Scheme work in progress*/
void TDMA(double a[], double b[], double c[], double d[], int n)
{/*  
Solves a system with tridiagonal matrix by LU factorization (diag(L) = 1).
a - lower codiagonal (i=2,n)
b - main diagonal (i=1,n)
c - upper codiagonal (i=1,n-1)
 d - constant terms (i=1,n); solution on exit
n - order of system.
*/
    int i;
    if (b[1] == 0e0) 
        { printf("TriDiagSys: singular matrix !\n"); return; }
    for (i=2; i<=n; i++) 
       {                                    // factorization
        a[i] /= b[i-1];
        b[i] -= a[i]*c[i-1];
        if (b[i] == 0e0) 
	    { printf("TriDiagSys: singular matrix !\n"); return; }
        d[i] -= a[i]*d[i-1];
       }
    d[n] /= b[n];                                     // backward substitution
    for (i=(n-1); i>=1; i--) 
        d[i] = (d[i] - c[i]*d[i+1])/b[i];
}

/*int main()
{
   double *a, *b, *c, *d;
   int i, n;

   n = 4;                                                  // order of system
   a = Vector(1,n);                                         // lower diagonal
   b = Vector(1,n);                                          // main diagonal
   c = Vector(1,n);                                         // upper diagonal
   d = Vector(1,n);                            // constant terms and solution

   a[1] = 0; b[1] = 1; c[1] = 2; d[1] = 1;
   a[2] = 2; b[2] = 1; c[2] = 2; d[2] = 2;
   a[3] = 2; b[3] = 1; c[3] = 2; d[3] = 3;
   a[4] = 2; b[4] = 1; c[4] = 0; d[4] = 4;  // Solution: -3.0, 2.0, 3.0, -2.0

   TDMA(a,b,c,d,n);                         // solve tridiagonal system

   printf("Solution:\n");
   for (i=1; i<=n; i++) printf("%10.3f",d[i]);
   printf("\n");
}
*/
